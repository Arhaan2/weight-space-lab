"""Pinned, offline native generation adapters for the acquired Qwen architecture.

Heavy inference libraries are imported only when a real backend is constructed.
The Transformers path is an explicit numerical/runtime variation from vLLM.
Neither adapter grades a generation or downloads a checkpoint.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import math
import os
import platform
import re
import resource
import sys
import time
from pathlib import Path

from .inventory import read_json


TRANSFORMERS_VERSIONS = {
    "transformers": "5.3.0", "torch": "2.10.0", "tokenizers": "0.22.2",
    "safetensors": "0.7.0", "jinja2": "3.1.6",
}
VLLM_VERSIONS = {**TRANSFORMERS_VERSIONS, "vllm": "0.17.0", "transformers": "4.57.6"}
ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
# The pinned ordinary autoregressive model does not instantiate the auxiliary
# one-layer multi-token prediction head. Keep these stored assets and report
# their identities; never treat a missing ordinary model weight as auxiliary.
MTP_TENSORS = frozenset({
    "mtp.fc.weight", "mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight", "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight", "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.self_attn.k_norm.weight", "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight", "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight", "mtp.layers.0.self_attn.v_proj.weight",
})


class NativeBackendError(ValueError):
    """A stable, publishable failure code without local paths/provider text."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} has missing or unsupported settings")


def _number(value, label, low, high):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{label} is outside its supported finite range")


def validate_backend_spec(spec):
    """Reject unsupported controls before loading any model or optional library."""
    _keys(spec, {"name", "versions", "device", "precision", "attention_implementation",
                 "max_model_len", "cpu_threads", "sampling", "chat", "stopping", "decoding"}, "backend")
    name = spec["name"]
    if name not in {"vllm", "transformers"}:
        raise ValueError("native backend must be vllm or transformers")
    versions = VLLM_VERSIONS if name == "vllm" else TRANSFORMERS_VERSIONS
    if spec["versions"] != versions:
        raise ValueError("backend versions must match the supported dependency pins")
    if spec["precision"] != "bfloat16":
        raise ValueError("only the declared bfloat16 execution precision is supported")
    expected = ("cuda", "auto") if name == "vllm" else ("cpu", "eager")
    if (spec["device"], spec["attention_implementation"]) != expected:
        raise ValueError("unsupported backend device or attention implementation")
    for key, low, high in (("max_model_len", 1, 262144), ("cpu_threads", 1, 64)):
        if type(spec[key]) is not int or not low <= spec[key] <= high:
            raise ValueError(f"{key} must be an integer within the supported bound")
    sampling = spec["sampling"]
    _keys(sampling, {"mode", "temperature", "top_p", "top_k", "min_p", "presence_penalty",
                     "frequency_penalty", "repetition_penalty", "max_new_tokens"}, "sampling")
    if sampling["mode"] not in {"stochastic", "greedy"}:
        raise ValueError("sampling mode must be explicit")
    _number(sampling["temperature"], "temperature", 0, 100)
    _number(sampling["top_p"], "top_p", 0, 1)
    if sampling["top_p"] == 0:
        raise ValueError("top_p must be positive")
    _number(sampling["presence_penalty"], "presence_penalty", -2, 2)
    for key, required in (("min_p", 0), ("frequency_penalty", 0), ("repetition_penalty", 1)):
        _number(sampling[key], key, required, required)
    if type(sampling["top_k"]) is not int or not 0 <= sampling["top_k"] <= 248320:
        raise ValueError("top_k must be a nonnegative integer within the model vocabulary")
    if type(sampling["max_new_tokens"]) is not int or not 1 <= sampling["max_new_tokens"] < spec["max_model_len"]:
        raise ValueError("output limit must be positive and below the explicit context limit")
    if sampling["mode"] == "greedy":
        if (sampling["temperature"], sampling["top_p"], sampling["top_k"]) != (0, 1, 0):
            raise ValueError("greedy requires temperature 0, top_p 1, top_k 0")
    elif sampling["temperature"] < 0.01:
        # vLLM clamps smaller positive temperatures; never silently accept that.
        raise ValueError("stochastic temperature must be at least 0.01")
    chat = spec["chat"]
    _keys(chat, {"enable_thinking", "add_generation_prompt", "chat_template_sha256"}, "chat")
    if type(chat["enable_thinking"]) is not bool or chat["add_generation_prompt"] is not True:
        raise ValueError("thinking must be boolean and add_generation_prompt must be true")
    if not isinstance(chat["chat_template_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", chat["chat_template_sha256"]) is None:
        raise ValueError("chat template must have an explicit SHA256")
    stopping = spec["stopping"]
    _keys(stopping, {"eos_token_ids", "stop_strings", "ignore_eos", "include_stop_str_in_output"}, "stopping")
    ids = stopping["eos_token_ids"]
    if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("EOS token IDs must be an explicit nonempty unique integer list")
    if stopping["stop_strings"] != [] or stopping["ignore_eos"] is not False or stopping["include_stop_str_in_output"] is not False:
        raise ValueError("only explicit EOS and length stopping without stop strings is supported")
    _keys(spec["decoding"], {"skip_special_tokens", "clean_up_tokenization_spaces"}, "decoding")
    if type(spec["decoding"]["skip_special_tokens"]) is not bool or spec["decoding"]["clean_up_tokenization_spaces"] is not False:
        raise ValueError("decoding requires an explicit special-token policy and no text cleanup")
    return copy.deepcopy(spec)


def _check_versions(versions):
    for package, required in versions.items():
        try:
            observed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise NativeBackendError("runtime_dependency_unavailable") from error
        if observed != required:
            raise NativeBackendError("runtime_version_mismatch")


def peak_rss_bytes():
    """Process high-water RSS, not system memory or a GPU allocation estimate."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _hardware():
    return {"system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "python_version": platform.python_version(),
            "logical_cpu_count": os.cpu_count()}


def _load_tokenizer(model_path, spec):
    from transformers import AutoTokenizer
    config = read_json(model_path / "config.json")
    if config.get("architectures") != [ARCHITECTURE] or config.get("model_type") != "qwen3_5":
        raise NativeBackendError("unsupported_checkpoint_architecture")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    template = (model_path / "chat_template.jinja").read_text(encoding="utf-8")
    if hashlib.sha256(template.encode()).hexdigest() != spec["chat"]["chat_template_sha256"]:
        raise NativeBackendError("chat_template_mismatch")
    if tokenizer.get_chat_template() != template:
        raise NativeBackendError("loaded_chat_template_mismatch")
    if not tokenizer.is_fast:
        raise NativeBackendError("unexpected_tokenizer_implementation")
    expected_eos = {config["text_config"]["eos_token_id"], tokenizer.eos_token_id}
    if set(spec["stopping"]["eos_token_ids"]) != expected_eos:
        raise NativeBackendError("eos_configuration_mismatch")
    return tokenizer, config, template


def _prompt(tokenizer, template, spec, messages, seed):
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise NativeBackendError("unsupported_seed")
    if not isinstance(messages, list) or not messages:
        raise NativeBackendError("invalid_messages")
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in {"system", "user", "assistant"}
                or not isinstance(message["content"], str)):
            raise NativeBackendError("unsupported_message_format")
    rendered = tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=False,
        add_generation_prompt=spec["chat"]["add_generation_prompt"],
        enable_thinking=spec["chat"]["enable_thinking"])
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not ids or len(ids) + spec["sampling"]["max_new_tokens"] > spec["max_model_len"]:
        raise NativeBackendError("context_limit_exceeded")
    return ids, rendered


def apply_presence_penalty(scores, generated_ids, penalty):
    """Subtract once per generated token type; exclude prompt and frequency.

    Operates on a single vocabulary score vector and a sequence of integer IDs;
    the same function supports dependency-free fixtures and Torch tensors.
    """
    for token in set(generated_ids):
        scores[token] -= penalty
    return scores


class _PresencePenalty:
    def __init__(self, prompt_length, penalty):
        self.prompt_length, self.penalty = prompt_length, penalty

    def __call__(self, input_ids, scores):
        for index in range(input_ids.shape[0]):
            apply_presence_penalty(scores[index], input_ids[index, self.prompt_length:].tolist(), self.penalty)
        return scores


def _result(tokenizer, spec, input_ids, output_ids, duration, rendered, *, reason=None):
    if not output_ids:
        raise NativeBackendError("empty_generation")
    last_is_eos = output_ids[-1] in spec["stopping"]["eos_token_ids"]
    if reason is None:
        reason = "eos" if last_is_eos else "length"
    if reason == "length" and len(output_ids) != spec["sampling"]["max_new_tokens"]:
        raise NativeBackendError("unexpected_early_termination")
    if reason == "eos" and not last_is_eos:
        raise NativeBackendError("inconsistent_finish_reason")
    text = tokenizer.decode(output_ids, **spec["decoding"])
    outcome = {"status": "truncated" if reason == "length" else "ok", "raw_output": text,
               "finish_reason": reason, "token_counts": {"input": len(input_ids), "output": len(output_ids),
                                                           "total": len(input_ids) + len(output_ids)},
               "duration_seconds": duration, "error": None}
    return {"outcome": outcome, "metadata": {
        "prompt_token_ids": list(input_ids), "output_token_ids": list(output_ids),
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "chat_template_sha256": spec["chat"]["chat_template_sha256"],
        "process_peak_rss_bytes": peak_rss_bytes(), "scientific_score": None,
        "duration_scope": "prompt_rendering_tokenization_and_generation_excluding_load_and_decode",
    }}


def _validate_runtime_tensor_names(stored_names, runtime_names):
    stored, runtime = set(stored_names), set(runtime_names)
    missing, extra = stored - runtime, runtime - stored
    if missing != MTP_TENSORS or extra != {"lm_head.weight"}:
        raise NativeBackendError("loaded_tensor_layout_mismatch")
    return sorted(missing)


def _preserve_and_verify_loaded_tensors(model, model_path):
    """Verify active saved tensors, the tied head, and explicitly report MTP.

    Generic dtype loading may cast source F32 norm/state parameters. Restore
    those source values before any forward pass, retaining the BF16 weights.
    The 15 auxiliary MTP tensors are retained on disk and hashed in metadata;
    the pinned ordinary autoregressive class deliberately does not load them.
    """
    import torch
    from safetensors import safe_open
    index = read_json(model_path / "model.safetensors.index.json")
    mapping = index["weight_map"]
    state = model.state_dict()
    omitted = _validate_runtime_tensor_names(mapping, state)
    head, embedding = state["lm_head.weight"], state["model.language_model.embed_tokens.weight"]
    if (not model.config.tie_word_embeddings
            or head.data_ptr() != embedding.data_ptr() or head.dtype != embedding.dtype
            or head.shape != embedding.shape or head.stride() != embedding.stride()):
        raise NativeBackendError("loaded_tied_embedding_mismatch")
    restored, unused, checked = [], [], 0
    for shard in sorted(set(mapping.values())):
        with safe_open(str(model_path / shard), framework="pt", device="cpu") as source:
            for name in source.keys():
                saved = source.get_tensor(name)
                if name in omitted:
                    unused.append({"name": name, "dtype": str(saved.dtype), "shape": list(saved.shape),
                                   "sha256": hashlib.sha256(saved.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                                   "reason": "auxiliary_mtp_disabled_for_ordinary_autoregression"})
                    continue
                if name not in state:
                    raise NativeBackendError("loaded_tensor_missing")
                runtime = state[name]
                if saved.dtype == torch.float32 and runtime.dtype != torch.float32:
                    try:
                        parameter = model.get_parameter(name)
                    except AttributeError:
                        parameter = model.get_buffer(name)
                    parameter.data = saved.clone()
                    runtime = parameter
                    restored.append(name)
                if (runtime.dtype != saved.dtype or runtime.shape != saved.shape
                        or not torch.equal(runtime.detach().cpu().contiguous().view(torch.uint8),
                                           saved.contiguous().view(torch.uint8))):
                    raise NativeBackendError("loaded_tensor_mismatch")
                checked += 1
    return {"verified_stored_tensor_count": checked, "restored_f32_tensors": restored,
            "unused_stored_tensors": sorted(unused, key=lambda item: item["name"]),
            "multi_token_prediction": False, "verified_tied_output_head": True,
            "precision_policy": "stored_bf16_and_f32_preserved", "runtime_parameter_dtypes": _dtype_counts(model)}


def _dtype_counts(model):
    counts = {}
    for parameter in model.parameters():
        dtype = str(parameter.dtype)
        counts[dtype] = counts.get(dtype, 0) + parameter.numel()
    return counts


class TransformersBackend:
    def __init__(self, spec, model_path):
        started = time.perf_counter()
        import torch
        from transformers import Qwen3_5ForConditionalGeneration
        self.spec, self.torch = spec, torch
        torch.set_num_threads(spec["cpu_threads"])
        # This path is intentionally CPU-only; no automatic device fallback.
        self.tokenizer, config, self.template = _load_tokenizer(model_path, spec)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation="eager")
        if type(self.model).__name__ != ARCHITECTURE:
            raise NativeBackendError("loaded_architecture_mismatch")
        self.model.eval()
        self.model.requires_grad_(False)
        tensors = _preserve_and_verify_loaded_tensors(self.model, model_path)
        self.metadata = {"backend": "transformers", "versions": spec["versions"],
                         "architecture": type(self.model).__name__, "device": "cpu",
                         "attention_implementation": "eager", "hardware": _hardware(),
                         "cpu_threads": torch.get_num_threads(), "load_seconds": time.perf_counter() - started,
                         "process_peak_rss_bytes": peak_rss_bytes(), "evidence_kind": "real_model",
                         "methodological_variation": "transformers_cpu_checkpoint_preserving_bf16_f32",
                         "state_accumulator_dtype": config["text_config"].get("mamba_ssm_dtype"),
                         "tokenizer_class": type(self.tokenizer).__name__,
                         "chat_template_sha256": spec["chat"]["chat_template_sha256"], **tensors}

    def generate(self, messages, seed):
        from transformers import GenerationConfig, LogitsProcessorList
        started = time.perf_counter()
        ids, rendered = _prompt(self.tokenizer, self.template, self.spec, messages, seed)
        torch, sampling = self.torch, self.spec["sampling"]
        torch.manual_seed(seed)
        options = {
            "do_sample": sampling["mode"] == "stochastic", "max_new_tokens": sampling["max_new_tokens"],
            "num_beams": 1, "num_return_sequences": 1, "use_cache": True,
            "eos_token_id": self.spec["stopping"]["eos_token_ids"],
            "pad_token_id": self.tokenizer.pad_token_id, "bos_token_id": None,
            "repetition_penalty": 1.0, "renormalize_logits": False,
        }
        if sampling["mode"] == "stochastic":
            options.update(temperature=sampling["temperature"], top_p=sampling["top_p"], top_k=sampling["top_k"])
        config = GenerationConfig(**options)
        encoded = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=encoded, attention_mask=torch.ones_like(encoded), generation_config=config,
                logits_processor=LogitsProcessorList([_PresencePenalty(len(ids), sampling["presence_penalty"])]))
        output_ids = generated[0, len(ids):].tolist()
        return _result(self.tokenizer, self.spec, ids, output_ids, time.perf_counter() - started, rendered)


def vllm_sampling_kwargs(spec, seed):
    """A directly testable, explicit mapping; no penalty substitutions."""
    sampling, stopping = spec["sampling"], spec["stopping"]
    return {"n": 1, "seed": seed, "temperature": sampling["temperature"],
            "top_p": sampling["top_p"], "top_k": sampling["top_k"], "min_p": sampling["min_p"],
            "presence_penalty": sampling["presence_penalty"], "frequency_penalty": sampling["frequency_penalty"],
            "repetition_penalty": sampling["repetition_penalty"], "max_tokens": sampling["max_new_tokens"],
            "min_tokens": 0, "stop": stopping["stop_strings"], "stop_token_ids": stopping["eos_token_ids"],
            "ignore_eos": stopping["ignore_eos"], "include_stop_str_in_output": stopping["include_stop_str_in_output"],
            "detokenize": False}


def _vllm_model_summary(model):
    return {"architecture": type(model).__name__, "runtime_parameter_dtypes": _dtype_counts(model)}


class VllmBackend:
    def __init__(self, spec, model_path):
        started = time.perf_counter()
        import torch
        from vllm import LLM
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise NativeBackendError("cuda_bfloat16_unavailable")
        torch.set_num_threads(spec["cpu_threads"])
        self.spec = spec
        self.tokenizer, _, self.template = _load_tokenizer(model_path, spec)
        self.model = LLM(model=str(model_path), tokenizer=str(model_path), dtype="bfloat16",
                         trust_remote_code=False, max_model_len=spec["max_model_len"],
                         tensor_parallel_size=1, seed=0, generation_config="vllm",
                         enable_prefix_caching=False, max_num_seqs=1,
                         enforce_eager=True, gpu_memory_utilization=0.8, speculative_config=None)
        loaded = self.model.apply_model(_vllm_model_summary)
        if len(loaded) != 1 or loaded[0]["architecture"] != ARCHITECTURE:
            raise NativeBackendError("loaded_architecture_mismatch")
        self.metadata = {"backend": "vllm", "versions": spec["versions"], "device": "cuda",
                         "hardware": {**_hardware(), "cuda_version": torch.version.cuda,
                                      "gpu_name": torch.cuda.get_device_name(0)},
                         "load_seconds": time.perf_counter() - started,
                         "process_peak_rss_bytes": peak_rss_bytes(), "evidence_kind": "real_model",
                         "precision_policy": "vllm_native_bfloat16_with_native_f32_state",
                         "multi_token_prediction": False,
                         "methodological_variation": "full_checkpoint_serial_seeded_requests_pinned_runtime",
                         "runtime_engine_settings": {"tensor_parallel_size": 1, "seed": 0,
                             "enable_prefix_caching": False, "max_num_seqs": 1,
                             "enforce_eager": True, "gpu_memory_utilization": 0.8,
                             "speculative_config": None},
                         "tokenizer_class": type(self.tokenizer).__name__,
                         "chat_template_sha256": spec["chat"]["chat_template_sha256"], **loaded[0]}

    def generate(self, messages, seed):
        from vllm import SamplingParams
        started = time.perf_counter()
        ids, rendered = _prompt(self.tokenizer, self.template, self.spec, messages, seed)
        outputs = self.model.generate([{"prompt_token_ids": ids}],
                                      SamplingParams(**vllm_sampling_kwargs(self.spec, seed)), use_tqdm=False)
        if len(outputs) != 1 or len(outputs[0].outputs) != 1 or not outputs[0].finished:
            raise NativeBackendError("incomplete_backend_output")
        output = outputs[0].outputs[0]
        if list(outputs[0].prompt_token_ids) != ids:
            raise NativeBackendError("runtime_prompt_mismatch")
        if output.finish_reason not in {"stop", "length"}:
            raise NativeBackendError("unsupported_finish_reason")
        reason = "eos" if output.finish_reason == "stop" else "length"
        return _result(self.tokenizer, self.spec, ids, list(output.token_ids),
                       time.perf_counter() - started, rendered, reason=reason)


def create_backend(spec, model_path):
    checked = validate_backend_spec(spec)
    _check_versions(checked["versions"])
    # Offline flags also cover library internals. No remote code is permitted.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    backend = VllmBackend if checked["name"] == "vllm" else TransformersBackend
    return backend(checked, Path(model_path))
