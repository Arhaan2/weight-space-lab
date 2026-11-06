import copy
import importlib.metadata
import unittest
from unittest.mock import patch

from weight_space_lab.native_backend import (
    MTP_TENSORS, NativeBackendError, TRANSFORMERS_VERSIONS, VLLM_VERSIONS, _PresencePenalty,
    _check_versions, _prompt, _result, _validate_runtime_tensor_names,
    apply_presence_penalty, validate_backend_spec,
    vllm_sampling_kwargs,
)


def backend_spec(name="transformers"):
    return {
        "name": name, "versions": copy.deepcopy(VLLM_VERSIONS if name == "vllm" else TRANSFORMERS_VERSIONS),
        "device": "cuda" if name == "vllm" else "cpu", "precision": "bfloat16",
        "attention_implementation": "auto" if name == "vllm" else "eager",
        "max_model_len": 2048, "cpu_threads": 4,
        "sampling": {"mode": "stochastic", "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                     "min_p": 0.0, "presence_penalty": 1.5, "frequency_penalty": 0.0,
                     "repetition_penalty": 1.0, "max_new_tokens": 16},
        "chat": {"enable_thinking": True, "add_generation_prompt": True,
                 "chat_template_sha256": "1" * 64},
        "stopping": {"eos_token_ids": [8, 9], "stop_strings": [], "ignore_eos": False,
                     "include_stop_str_in_output": False},
        "decoding": {"skip_special_tokens": False, "clean_up_tokenization_spaces": False},
    }


class TokenizerFixture:
    def apply_chat_template(self, messages, **kwargs):
        self.template_arguments = kwargs
        return "synthetic rendered prompt"

    def encode(self, text, **kwargs):
        self.encode_arguments = kwargs
        return [1, 2, 3]

    def decode(self, ids, **kwargs):
        self.decode_arguments = kwargs
        return "synthetic ungraded output"


class NativeBackendContractTests(unittest.TestCase):
    def test_only_the_exact_auxiliary_mtp_layout_may_be_unused(self):
        active = {"model.language_model.embed_tokens.weight",
                  "model.language_model.layers.15.mlp.down_proj.weight"}
        saved, runtime = active | MTP_TENSORS, active | {"lm_head.weight"}
        self.assertEqual(_validate_runtime_tensor_names(saved, runtime), sorted(MTP_TENSORS))
        for changed_saved, changed_runtime in (
                (saved, runtime - {"model.language_model.layers.15.mlp.down_proj.weight"}),
                (saved | {"mtp.undeclared.weight"}, runtime),
                (saved - {"mtp.fc.weight"}, runtime),
                (saved, runtime | {"unexpected.weight"})):
            with self.assertRaisesRegex(NativeBackendError, "loaded_tensor_layout_mismatch"):
                _validate_runtime_tensor_names(changed_saved, changed_runtime)

    def test_explicit_pins_and_defensive_copy(self):
        for name in ("transformers", "vllm"):
            spec = backend_spec(name)
            checked = validate_backend_spec(spec)
            checked["sampling"]["presence_penalty"] = 0
            self.assertEqual(spec["sampling"]["presence_penalty"], 1.5)

    def test_unsupported_required_settings_are_rejected(self):
        changes = [
            ("sampling", "frequency_penalty", 0.5), ("sampling", "repetition_penalty", 1.5),
            ("sampling", "min_p", 0.1), ("sampling", "temperature", 0.001),
            ("sampling", "temperature", float("nan")), ("sampling", "top_k", True),
            ("sampling", "temperature", 10 ** 1000),
            ("sampling", "max_new_tokens", 2048), ("sampling", "top_p", 0),
            ("sampling", "presence_penalty", 4), ("sampling", "max_tokens", 16),
            ("stopping", "stop_strings", ["END"]), ("stopping", "ignore_eos", True),
            ("stopping", "eos_token_ids", [9, 9]), ("stopping", "eos_token_ids", [True]),
            ("decoding", "clean_up_tokenization_spaces", True), ("chat", "enable_thinking", None),
            ("chat", "add_generation_prompt", False), ("versions", "torch", "2.9.0"),
        ]
        for group, key, value in changes:
            with self.subTest(group=group, key=key, value=value):
                spec = backend_spec()
                spec[group][key] = value
                with self.assertRaises(ValueError):
                    validate_backend_spec(spec)
        for key, value in (("name", "automatic"), ("device", "mps"), ("precision", "float32"),
                           ("attention_implementation", "sdpa"), ("cpu_threads", 0)):
            spec = backend_spec()
            spec[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_backend_spec(spec)

    def test_greedy_settings_must_be_consistent(self):
        spec = backend_spec()
        spec["sampling"]["mode"] = "greedy"
        with self.assertRaisesRegex(ValueError, "greedy"):
            validate_backend_spec(spec)
        spec["sampling"].update(temperature=0, top_p=1, top_k=0)
        self.assertEqual(validate_backend_spec(spec), spec)

    def test_presence_is_additive_once_per_output_token(self):
        # Prompt contains 0, output contains 1 twice and 2 once. Prompt-only
        # tokens and frequency must not affect presence-penalty semantics.
        scores = [4.0, 3.0, -2.0, 0.0]
        observed = apply_presence_penalty(scores, [1, 1, 2], 1.5)
        self.assertEqual(observed, [4.0, 1.5, -3.5, 0.0])
        self.assertEqual(apply_presence_penalty([4.0], [], 1.5), [4.0])
        self.assertEqual(apply_presence_penalty([1.0], [0], -1.5), [2.5])

    def test_presence_processor_excludes_prompt_span(self):
        import numpy as np
        ids = np.array([[0, 0, 1, 1, 2]])
        scores = np.array([[4.0, 3.0, -2.0, 0.0]])
        result = _PresencePenalty(2, 1.5)(ids, scores)
        self.assertEqual(result.tolist(), [[4.0, 1.5, -3.5, 0.0]])

    def test_vllm_mapping_keeps_presence_and_repetition_separate(self):
        spec = backend_spec("vllm")
        kwargs = vllm_sampling_kwargs(spec, 17)
        self.assertEqual(kwargs["presence_penalty"], 1.5)
        self.assertEqual(kwargs["repetition_penalty"], 1.0)
        self.assertEqual(kwargs["frequency_penalty"], 0.0)
        self.assertEqual(kwargs["seed"], 17)
        self.assertEqual(kwargs["n"], 1)
        self.assertEqual(kwargs["max_tokens"], 16)
        self.assertEqual(kwargs["stop_token_ids"], [8, 9])
        self.assertFalse(kwargs["detokenize"])

    def test_prompt_uses_exact_template_and_does_not_add_special_tokens(self):
        tokenizer = TokenizerFixture()
        ids, text = _prompt(tokenizer, "fixture template", backend_spec(),
                            [{"role": "user", "content": "Synthetic question"}], 17)
        self.assertEqual(ids, [1, 2, 3])
        self.assertEqual(tokenizer.template_arguments, {"chat_template": "fixture template",
            "tokenize": False, "add_generation_prompt": True, "enable_thinking": True})
        self.assertEqual(tokenizer.encode_arguments, {"add_special_tokens": False})
        spec = backend_spec()
        spec["max_model_len"] = 18
        with self.assertRaisesRegex(NativeBackendError, "context_limit_exceeded"):
            _prompt(tokenizer, "fixture", spec, [{"role": "user", "content": "x"}], 17)
        with self.assertRaisesRegex(NativeBackendError, "unsupported_message"):
            _prompt(tokenizer, "fixture", spec, [{"role": "user", "content": []}], 17)

    def test_raw_output_tokens_and_finish_reasons(self):
        tokenizer, spec = TokenizerFixture(), backend_spec()
        eos = _result(tokenizer, spec, [1, 2, 3], [4, 9], 0.2, "fixture")
        self.assertEqual(eos["outcome"]["finish_reason"], "eos")
        self.assertEqual(eos["outcome"]["token_counts"], {"input": 3, "output": 2, "total": 5})
        self.assertEqual(eos["metadata"]["output_token_ids"], [4, 9])
        self.assertIsNone(eos["metadata"]["scientific_score"])
        self.assertEqual(tokenizer.decode_arguments, spec["decoding"])
        cut = _result(tokenizer, spec, [1], [4] * 16, 0.3, "fixture")
        self.assertEqual(cut["outcome"]["status"], "truncated")
        with self.assertRaisesRegex(NativeBackendError, "unexpected_early"):
            _result(tokenizer, spec, [1], [4] * 3, 0.3, "fixture")
        with self.assertRaisesRegex(NativeBackendError, "empty_generation"):
            _result(tokenizer, spec, [1], [], 0.3, "fixture")

    def test_version_mismatch_and_missing_dependency_fail_closed(self):
        with patch("importlib.metadata.version", return_value="0.0.0"):
            with self.assertRaisesRegex(NativeBackendError, "runtime_version_mismatch"):
                _check_versions(TRANSFORMERS_VERSIONS)
        with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
            with self.assertRaisesRegex(NativeBackendError, "runtime_dependency_unavailable"):
                _check_versions(TRANSFORMERS_VERSIONS)


if __name__ == "__main__":
    unittest.main()
