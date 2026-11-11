"""Diagnostic semantics and failure accounting, without any real model calls."""
import contextlib
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from test_checkpoint import checkpoint_fixture
from weight_space_lab.grading import ifeval_metrics
from weight_space_lab.inventory import inventory_model
from weight_space_lab.protocol import canonical_hash
from weight_space_lab.research_probes import (
    DEFAULT_LIMITS, _Arrays, _eos_ids, _expected_array_bytes, _generation_rows, _grade,
    _render_teacher, _run_worker, _teacher_one, paired_seed, paired_summary, run_probe,
    validate_limits, validate_request,
)


def item(identity="practice-1", split="development"):
    return {"benchmark": "ifeval", "item_id": identity, "dataset_revision": "a" * 40,
            "split": split, "task_text": "Answer with a bracketed placeholder.",
            "target": {"instruction_id_list": ["detectable_content:number_placeholders"], "kwargs": [{"num_placeholders": 1}]},
            "source": {"dataset_id": "test-fixture", "revision": "a" * 40, "split": "train", "row_id": identity}, "metadata": {}}


def teacher(**kw):
    return {"type": "teacher_forced", "model": "original", "texts": ["context"],
            "return": {"token_logprobs": True}, **kw}


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    def __getitem__(self, key):
        return Tensor(self.value[key])

    def __sub__(self, other):
        return Tensor(self.value - other.value)

    def gather(self, dim, indices):
        return Tensor(np.take_along_axis(self.value, indices.value, axis=dim))

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return Tensor(self.value.astype(np.float32))

    def numpy(self):
        return self.value

    def tolist(self):
        return self.value.tolist()


class Module:
    def __init__(self):
        self.pre, self.post = [], []

    def register_forward_pre_hook(self, fn):
        self.pre.append(fn)
        return SimpleNamespace(remove=lambda: self.pre.remove(fn))

    def register_forward_hook(self, fn):
        self.post.append(fn)
        return SimpleNamespace(remove=lambda: self.post.remove(fn))

    def emit(self, value):
        for fn in self.pre:
            fn(self, (value,))
        for fn in self.post:
            fn(self, (value,), value)


def fake_model():
    blocks = []
    for _ in range(24):
        block = Module()
        block.mlp = SimpleNamespace(down_proj=Module())
        block.self_attn = Module()
        blocks.append(block)
    embedding, norm = Module(), Module()

    class Backbone(Module):
        layers, embed_tokens = blocks, embedding

        def __call__(self, input_ids, **kwargs):
            for hook in self.pre:
                hook(self, ())
            n = input_ids.shape[1]
            value = Tensor(np.arange(n * 2).reshape(1, n, 2) / 4)
            embedding.emit(value)
            for block in blocks:
                block.mlp.down_proj.emit(value)
                block.emit(value)
                for hook in block.self_attn.post:
                    hook(block.self_attn, (), (value, Tensor(np.eye(n)[None, None])))
            norm.emit(value)
            result = SimpleNamespace(last_hidden_state=value)
            for hook in self.post:
                hook(self, (), result)
            return result
    backbone = Backbone()
    backbone.norm = norm
    weights = np.array([[1, 0, -1, 2], [0, 1, 1, -1]])
    model = SimpleNamespace(model=SimpleNamespace(language_model=backbone), eval=lambda: None,
                            requires_grad_=lambda enabled: None,
                            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2, intermediate_size=2, num_attention_heads=1)),
                            lm_head=lambda x: Tensor(x.value @ weights))
    def topk(x, k, dim):
        indices = np.argsort(-x.value, axis=dim)[..., :k]
        return Tensor(np.take_along_axis(x.value, indices, axis=dim)), Tensor(indices)
    torch = SimpleNamespace(tensor=lambda x, **kw: Tensor(x), long="long", bfloat16="bfloat16",
        set_num_threads=lambda n: None, inference_mode=contextlib.nullcontext,
        logsumexp=lambda x, dim, keepdim: Tensor(np.log(np.exp(x.value).sum(axis=dim, keepdims=keepdim))), topk=topk)
    return model, torch, [embedding, norm, *blocks, *[b.mlp.down_proj for b in blocks], *[b.self_attn for b in blocks]]


class ResearchProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_free_text_and_all_layer_probes_are_not_fixed_contrast_pairs(self):
        request = teacher(texts=["one arbitrary context"], **{"return": {
            "hidden_states": list(range(25)), "mlp_act": list(range(24)), "token_logprobs": True, "topk": 50}})
        checked = validate_request(request)
        self.assertEqual(checked["return"]["hidden_states"], list(range(25)))
        self.assertEqual(checked["texts"], request["texts"])
        self.assertNotIn("enable_thinking", request)
        self.assertGreater(validate_limits()["max_input_tokens"], 36)

    def test_attention_is_real_full_attention_only_and_bounded(self):
        valid = teacher(**{"return": {"attn_layers": [3, 23]}})
        validate_request(valid)
        for layers in ([0], [3, 3], [24]):
            with self.assertRaises(ValueError):
                validate_request(teacher(**{"return": {"attn_layers": layers}}))
        with self.assertRaisesRegex(ValueError, "attention text"):
            validate_request({**valid, "texts": ["a", "b", "c"]})

    def test_unknown_options_and_nonfinite_settings_are_rejected(self):
        for request in (teacher(extra=True), teacher(**{"return": {"attention_surrogate": True}}),
                        {"type": "generate", "model": "original", "prompts": ["x"], "settings": {"temperature": float("nan")}},
                        {"type": "generate", "model": "original", "prompts": ["x"], "settings": {"seed": True}}):
            with self.assertRaises(ValueError):
                validate_request(request)
        with self.assertRaises(ValueError):
            validate_limits({"cpu_threads": 100})

    def test_both_arms_count_toward_generation_allowance(self):
        request = {"type": "generate", "model": "both", "prompts": ["x"], "n": 256}
        self.assertEqual(validate_request(request)["n"], 256)
        with self.assertRaisesRegex(ValueError, "both arms count"):
            validate_request({**request, "n": 257})

    def test_practice_only_and_no_unavailable_or_test_items(self):
        request = {"type": "generate", "model": "both", "problems": ["practice-1"]}
        self.assertEqual(validate_request(request, practice_items=[item()])["problems"], ["practice-1"])
        with self.assertRaises(ValueError):
            validate_request(request)
        with self.assertRaisesRegex(ValueError, "non-test"):
            validate_request(request, practice_items=[item(split="test")])

    def test_teacher_tokenization_preserves_exact_template_and_never_truncates(self):
        calls = []
        tokenizer = SimpleNamespace(encode=lambda text, **kw: list(range(len(text))),
            apply_chat_template=lambda chat, **kw: calls.append(kw) or "rendered")
        request = validate_request({"type": "teacher_forced", "model": "both", "chats": [[{"role": "assistant", "content": "x"}]],
            "enable_thinking": True, "return": {"token_logprobs": True}})
        rows = _render_teacher(tokenizer, request, DEFAULT_LIMITS)
        self.assertEqual(rows[0]["token_ids"], list(range(8)))
        self.assertFalse(calls[0]["add_generation_prompt"])
        self.assertTrue(calls[0]["enable_thinking"])
        with self.assertRaisesRegex(ValueError, "both arms count"):
            _render_teacher(tokenizer, request, {**DEFAULT_LIMITS, "max_total_tokens": 15})
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            _render_teacher(tokenizer, request, {**DEFAULT_LIMITS, "max_input_tokens": 3})

    def test_generation_reserves_replay_and_uses_paired_candidate_independent_seeds(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: "rendered", encode=lambda *a, **k: [1, 2])
        request = validate_request({"type": "generate", "model": "both", "prompts": ["x"], "max_tokens": 10,
            "activations": {"layers": [0, 19], "mlp": True, "max_answer_tokens": 5}})
        rows = _generation_rows(tokenizer, request, {}, DEFAULT_LIMITS)
        self.assertEqual(rows[0]["seed"], paired_seed(42, rows[0]["item_id"], 0))
        self.assertNotEqual(rows[0]["seed"], paired_seed(42, rows[0]["item_id"], 1))
        with self.assertRaisesRegex(ValueError, "replay total"):
            _generation_rows(tokenizer, request, {}, {**DEFAULT_LIMITS, "max_generation_tokens": 37})

    def test_eos_accepts_scalar_list_and_missing_tokenizer_value_without_implicit_defaults(self):
        self.assertEqual(_eos_ids(None, [151643, 151645]), [151643, 151645])
        self.assertEqual(_eos_ids(151645, [151643, 151645]), [151643, 151645])
        for values in ((None, None), (True,), ([-1],)):
            with self.assertRaises(ValueError):
                _eos_ids(*values)

    def test_per_token_probabilities_align_next_token_and_hooks_observe_real_outputs(self):
        model, torch, modules = fake_model()
        returns = {"token_logprobs": True, "topk": 2, "hidden_states": [0, 19, 24], "mlp_act": [19], "attn_layers": [3]}
        writer = _Arrays(self.root, 100000)
        _teacher_one(model, [0, 1, 2], returns, writer, "probe", torch)
        logps = np.load(self.root / "probe-token_logprobs.npy")
        hidden = np.arange(6).reshape(3, 2) / 4
        logits = hidden[:2] @ np.array([[1, 0, -1, 2], [0, 1, 1, -1]])
        expected = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        np.testing.assert_allclose(logps, expected[np.arange(2), [1, 2]], rtol=1e-6)
        np.testing.assert_array_equal(np.load(self.root / "probe-hidden_24.npy"), hidden)
        self.assertEqual(np.load(self.root / "probe-attention_3.npy").shape, (1, 3, 3))
        self.assertEqual(np.load(self.root / "probe-topk_ids.npy").shape, (2, 2))
        self.assertTrue(all(not m.pre and not m.post for m in modules))

    def test_hook_cleanup_even_when_artifact_budget_fails(self):
        model, torch, modules = fake_model()
        returns = {"token_logprobs": False, "topk": 0, "hidden_states": [0], "mlp_act": [], "attn_layers": []}
        with self.assertRaisesRegex(ValueError, "array byte"):
            _teacher_one(model, [0, 1, 2], returns, _Arrays(self.root, 1), "probe", torch)
        self.assertTrue(all(not m.pre and not m.post for m in modules))

    def test_worker_paired_teacher_uses_both_snapshots_and_counts_model_calls(self):
        original = checkpoint_fixture(self.root)
        edited = self.root / "edited"
        shutil.copytree(original, edited)
        output = self.root / "out"
        output.mkdir()
        loaded = []
        model, torch, _ = fake_model()
        def load(path, **kw):
            loaded.append(path)
            return fake_model()[0]
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(encode=lambda *a, **kw: [0, 1, 2])),
            Qwen3_5ForConditionalGeneration=SimpleNamespace(from_pretrained=load))
        request = validate_request(teacher(model="both", **{"return": {"token_logprobs": True, "hidden_states": [0, 24], "mlp_act": [19]}}))
        identity = canonical_hash(inventory_model(original)["files"])
        payload = {"original": str(original), "edited": str(edited), "original_sha256": identity,
                   "edited_sha256": identity, "limits": DEFAULT_LIMITS, "request": request,
                   "practice_items": [], "official_ifeval_root": None}
        with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}), \
             patch("weight_space_lab.native_backend._check_versions"), \
             patch("weight_space_lab.native_backend._preserve_and_verify_loaded_tensors", return_value={"verified": True}):
            _run_worker(payload, output)
        self.assertEqual(loaded, [str(original), str(edited)])
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["sha256"], canonical_hash(manifest["payload"]))
        counts = manifest["payload"]["accounting"]
        self.assertEqual(counts["model_forward_calls_completed"], 2)
        self.assertEqual(counts["teacher_forced_sequences_completed"], 2)
        self.assertEqual(counts["generation_sequences_started"], 0)
        self.assertEqual(len(manifest["payload"]["arrays"]), 8)

    def generation_runtime(self):
        _, torch, _ = fake_model()
        seeds, controls = [], []
        torch.manual_seed = seeds.append
        torch.ones = lambda shape, **kwargs: Tensor(np.ones(shape))
        tokenizer = SimpleNamespace(encode=lambda *a, **kw: [0, 1],
            apply_chat_template=lambda *a, **kw: "rendered prompt", eos_token_id=None, pad_token_id=None,
            decode=lambda *a, **kw: "[answer]<|im_end|>")
        def load(path, **kwargs):
            model, _, _ = fake_model()
            model.config.text_config.eos_token_id = [3]
            def generate(**kwargs):
                controls.append(kwargs)
                model.model.language_model(input_ids=kwargs["input_ids"])
                return Tensor(np.concatenate([kwargs["input_ids"].value, [[2, 3]]], axis=1))
            model.generate = generate
            return model
        transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
            Qwen3_5ForConditionalGeneration=SimpleNamespace(from_pretrained=load),
            GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs), LogitsProcessorList=list)
        return torch, transformers, seeds, controls

    def test_generation_worker_pairs_seeds_and_explicitly_counts_activation_replay(self):
        original = checkpoint_fixture(self.root)
        edited, output = self.root / "edited", self.root / "out"
        shutil.copytree(original, edited)
        output.mkdir()
        torch, transformers, seeds, controls = self.generation_runtime()
        request = validate_request({"type": "generate", "model": "both", "prompts": ["x"], "max_tokens": 4,
                                    "activations": {"layers": [0, 24], "mlp": True}})
        identity = canonical_hash(inventory_model(original)["files"])
        payload = {"original": str(original), "edited": str(edited), "original_sha256": identity,
                   "edited_sha256": identity, "limits": DEFAULT_LIMITS, "request": request,
                   "practice_items": [], "official_ifeval_root": None}
        with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}), \
             patch("weight_space_lab.native_backend._check_versions"), \
             patch("weight_space_lab.native_backend._preserve_and_verify_loaded_tensors", return_value={}):
            _run_worker(payload, output)
        manifest = json.loads((output / "manifest.json").read_text())["payload"]
        self.assertEqual(seeds, [seeds[0], seeds[0]])
        self.assertEqual(controls[0]["generation_config"].eos_token_id, [3])
        self.assertFalse(controls[0]["generation_config"].do_sample)
        self.assertEqual(controls[0]["logits_to_keep"], 1)
        self.assertEqual(manifest["accounting"]["model_forward_calls_completed"], 4)
        self.assertEqual(manifest["accounting"]["generation_sequences_completed"], 2)
        self.assertEqual(manifest["accounting"]["activation_replay_sequences_completed"], 2)
        self.assertEqual(manifest["accounting"]["generated_tokens"], 4)
        self.assertEqual(manifest["summaries"]["paired"]["graded_pairs"], 0)
        rows = json.loads((output / "original-generations.json").read_text())
        self.assertEqual(rows[0]["activation_replay"]["answer_tokens"], 2)

    def test_unexpected_grader_exception_retains_completed_raw_generation(self):
        original, output = checkpoint_fixture(self.root), self.root / "out"
        output.mkdir()
        torch, transformers, _, _ = self.generation_runtime()
        request = validate_request({"type": "generate", "model": "original", "problems": ["practice-1"], "max_tokens": 4}, practice_items=[item()])
        identity = canonical_hash(inventory_model(original)["files"])
        payload = {"original": str(original), "edited": None, "original_sha256": identity,
                   "edited_sha256": None, "limits": DEFAULT_LIMITS, "request": request,
                   "practice_items": [item()], "official_ifeval_root": "/fixture-official"}
        with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}), \
             patch("weight_space_lab.native_backend._check_versions"), \
             patch("weight_space_lab.native_backend._preserve_and_verify_loaded_tensors", return_value={}), \
             patch("weight_space_lab.research_probes._grade", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                _run_worker(payload, output)
        rows = json.loads((output / "original-generations.json").read_text())
        self.assertEqual(rows[0]["raw_output"], "[answer]<|im_end|>")
        self.assertIsNone(rows[0]["grade"])
        self.assertEqual(json.loads((output / "progress.json").read_text())["generation_sequences_completed"], 1)

    def test_attention_preflight_includes_quadratic_output_storage(self):
        count = _expected_array_bytes([{"token_ids": list(range(100))}],
            {"hidden_states": [], "mlp_act": [], "attn_layers": [3], "token_logprobs": False, "topk": 0},
            SimpleNamespace(hidden_size=1024, intermediate_size=3584, num_attention_heads=8))
        self.assertEqual(count, 4 * 8 * 100 * 100)

    def test_paired_grader_errors_are_null_and_do_not_become_failures(self):
        def row(i, correct):
            return {"item_id": str(i), "sample_index": 0, "seed": i,
                    "grade": {"status": "grader_error" if correct is None else "graded", "correct": correct}}
        result = paired_summary([row(1, False), row(2, None)], [row(1, True), row(2, True)])
        self.assertEqual(result["graded_pairs"], 1)
        self.assertEqual(result["ungraded_pairs"], 1)
        self.assertIsNone(result["paired_standard_error"])
        self.assertEqual(result["wrong_to_right"], 1)
        with self.assertRaises(ValueError):
            paired_summary([row(1, False)], [row(2, True)])

    def test_practice_grading_uses_official_bound_record_and_retains_truncation(self):
        row = {"sample_index": 0, "seed": 42, "raw_output": "[name]", "truncated": True,
               "duration_seconds": 0.1, "token_counts": {"input": 2, "output": 3, "total": 5}}
        request = validate_request({"type": "generate", "model": "original", "problems": ["practice-1"]}, practice_items=[item()])
        with patch("weight_space_lab.grading._worker", return_value={"status": "graded", "correct": True,
                   "metrics": ifeval_metrics([True], [True]), "code": "official_instruction_results"}) as worker:
            grade = _grade(row, item(), "b" * 64, request, "/official")
        self.assertTrue(grade["correct"])
        self.assertEqual(worker.call_args.args[0]["benchmark"], "ifeval")
        self.assertEqual(worker.call_args.args[0]["ifeval_root"], "/official")

    def test_supervisor_retains_failed_worker_evidence_and_does_not_retry(self):
        model = checkpoint_fixture(self.root)
        process = SimpleNamespace(returncode=1, pid=99999, poll=lambda: 1)
        output = self.root / "failed"
        with patch("weight_space_lab.research_probes.subprocess.Popen", return_value=process) as popen:
            with self.assertRaisesRegex(RuntimeError, "retained"):
                run_probe(model, None, teacher(), output)
        self.assertEqual(popen.call_count, 1)
        self.assertTrue((output / "worker.stderr").exists())
        self.assertEqual(json.loads((output / "failure.json").read_text())["payload"]["status"], "failed")
        self.assertTrue(json.loads((output / "supervision.json").read_text())["payload"]["process_exit_confirmed"])

    def test_timeout_closes_lifeline_and_waits_for_guardian(self):
        model = checkpoint_fixture(self.root)
        class Process:
            returncode = None
            pid = 99999
            def poll(self):
                return self.returncode
            def wait(self, **kw):
                self.returncode = -9
                return self.returncode
        output = self.root / "timeout"
        process = Process()
        with patch("weight_space_lab.research_probes.subprocess.Popen", return_value=process) as popen, \
             patch("weight_space_lab.research_probes.time.monotonic", side_effect=[0, 2, 3]):
            with self.assertRaises(TimeoutError):
                run_probe(model, None, teacher(), output, {"wall_seconds": 1})
        self.assertEqual(process.returncode, -9)
        self.assertTrue(popen.call_args.kwargs["pass_fds"])
        for descriptor in popen.call_args.kwargs["pass_fds"]:
            with self.assertRaises(OSError):
                os.fstat(descriptor)
        self.assertTrue((output / "supervision.json").is_file())

    def test_rss_supervision_measures_worker_not_guardian(self):
        model, output = checkpoint_fixture(self.root), self.root / "failed"
        class Process:
            pid, returncode, polls = 99999, None, 0
            def poll(self):
                self.polls += 1
                if self.polls > 1:
                    self.returncode = 1
                return self.returncode
        process = Process()
        def launch(*args, **kwargs):
            (output / "worker-identity.json").write_text(json.dumps({"pid": 88888, "parent_pid": 99999, "process_group": 88888}))
            return process
        with patch("weight_space_lab.research_probes.subprocess.Popen", side_effect=launch), \
             patch("weight_space_lab.mechanistic_tools._rss_bytes", return_value=12345) as rss, \
             patch("weight_space_lab.research_probes.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "retained"):
                run_probe(model, None, teacher(), output)
        rss.assert_called_once_with(88888)
        receipt = json.loads((output / "supervision.json").read_text())["payload"]
        self.assertEqual(receipt["sampled_worker_peak_rss_bytes"], 12345)
        self.assertEqual(receipt["watchdog_pid"], 99999)

    @unittest.skipUnless(os.name == "posix", "process groups require POSIX")
    def test_controller_sigkill_watchdog_reaps_worker_and_stops_descendant(self):
        model, output = checkpoint_fixture(self.root), self.root / "crashed-controller"
        # Replace only the trusted worker entry with a stdlib sleeping fixture.
        # This exercises run_probe's actual guardian/lifeline launch, no model.
        bootstrap = '''import json, os, subprocess, sys, time
output = sys.argv[2]
descendant = subprocess.Popen([sys.executable, "-I", "-S", "-c", "import time; time.sleep(60)"])
temporary = os.path.join(output, ".worker-identity.tmp")
with open(temporary, "x") as stream:
    json.dump({"pid": os.getpid(), "parent_pid": os.getppid(), "process_group": os.getpgrp(), "descendant": descendant.pid}, stream)
os.replace(temporary, os.path.join(output, "worker-identity.json"))
time.sleep(60)
'''
        code = ("import sys\nfrom weight_space_lab import research_probes as probe\n"
                + "from weight_space_lab import mechanistic_tools\nmechanistic_tools._rss_bytes = lambda pid: 1\n"
                + "probe._WORKER_BOOTSTRAP = " + repr(bootstrap) + "\n"
                + "probe.run_probe(sys.argv[1], None, " + repr(teacher())
                + ", sys.argv[2], {'wall_seconds': 30})\n")
        controller = subprocess.Popen([sys.executable, "-c", code, str(model), str(output)],
            cwd=Path(__file__).resolve().parents[1], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            start_new_session=True)
        identity = None
        def running(pid):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            return True
        try:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and controller.poll() is None:
                path = output / "worker-identity.json"
                if path.is_file():
                    identity = json.loads(path.read_text())
                    break
                time.sleep(0.05)
            if identity is None:
                controller.kill()
                _, stderr = controller.communicate(timeout=3)
                self.fail("synthetic worker never became ready: " + stderr.decode(errors="replace"))
            self.assertTrue(running(identity["pid"]))
            controller.kill()
            controller.wait(timeout=3)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and any(running(identity[key]) for key in ("pid", "descendant")):
                time.sleep(0.05)
            self.assertFalse(running(identity["pid"]), "watchdog must reap model worker after controller death")
            self.assertFalse(running(identity["descendant"]), "watchdog must kill all members of worker process group")
        finally:
            if controller.poll() is None:
                controller.kill()
                controller.wait(timeout=3)
            if identity:
                for key in ("pid", "parent_pid"):
                    try:
                        os.killpg(identity[key], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            if controller.stderr:
                controller.stderr.close()

    def test_edited_controls_or_source_identity_mismatch_rejected_before_worker(self):
        model = checkpoint_fixture(self.root)
        candidate = self.root / "candidate"
        shutil.copytree(model, candidate)
        (candidate / "config.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "controls"):
            run_probe(model, candidate, teacher(model="edited"), self.root / "probe")
        with self.assertRaisesRegex(ValueError, "identity"):
            run_probe(model, None, teacher(), self.root / "probe", expected_original_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
