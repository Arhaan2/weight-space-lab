"""Meterable original/edited diagnostic forwards, separate from final WebLLM runs.

No inference dependency is imported by validation or preparation. A disposable,
offline CPU worker runs an explicitly requested generation or teacher-forced
probe. Arbitrary texts, layers and comparisons are allowed: this module does not
prescribe a contrast, ranking, edit family or hypothesis. Hidden state 0 is the
embedding, 1..23 are decoder block outputs, and 24 is the final normalized state,
matching Transformers' hidden-state convention. MLP activations are gated inputs
to down_proj; attention is supported only for the six full-attention blocks.

All arrays are per-text, unpadded .npy files. Token log-probability index i is the
probability of input token i+1; the unconditioned first token has no entry.
Generation activation requests explicitly replay the prompt and answer prefix;
those extra forwards are accounted for, never described as free observations.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .inventory import file_sha256, inventory_model, read_json
from .protocol import canonical_hash


FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23)
DEFAULT_LIMITS = {
    "max_texts": 1024, "max_generations": 512, "max_input_tokens": 32768,
    "max_new_tokens": 32768, "max_total_tokens": 524288, "max_generation_tokens": 1048576,
    "max_context_tokens": 65536, "max_array_bytes": 6 * 1024 ** 3,
    "max_input_bytes": 16 * 1024 ** 2, "max_checkpoint_bytes": 4 * 1024 ** 3,
    "max_rss_bytes": 8 * 1024 ** 3, "wall_seconds": 3600,
    "max_attention_texts": 2, "max_attention_tokens": 2048,
    "max_log_bytes": 1024 ** 2, "cpu_threads": 4,
}
_MAX_LIMITS = {**DEFAULT_LIMITS, "max_rss_bytes": 96 * 1024 ** 3,
               "wall_seconds": 7200, "cpu_threads": 64, "max_generation_tokens": 67108864}
DEFAULT_SETTINGS = {"temperature": 0, "top_p": 1, "top_k": 0,
                    "presence_penalty": 0, "enable_thinking": False, "seed": 42}

# Runs before importing inference dependencies, inside the watchdog-owned worker
# process group. It identifies the worker independently of the guardian PID so
# RSS supervision cannot accidentally measure the tiny watchdog instead.
_WORKER_BOOTSTRAP = r'''
import json, os, runpy, sys
repository, output = sys.argv[1:3]
temporary = os.path.join(output, ".worker-identity.tmp")
with open(temporary, "x") as stream:
    json.dump({"pid": os.getpid(), "parent_pid": os.getppid(), "process_group": os.getpgrp()}, stream)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, os.path.join(output, "worker-identity.json"))
sys.path.insert(0, repository)
sys.argv = ["weight_space_lab.research_probes", "--worker", output]
runpy.run_module("weight_space_lab.research_probes", run_name="__main__")
'''


def _int(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


def validate_limits(limits=None):
    if limits is not None and (not isinstance(limits, dict) or set(limits) - set(DEFAULT_LIMITS)):
        raise ValueError("unknown probe limits")
    value = {**DEFAULT_LIMITS, **(limits or {})}
    for key, maximum in _MAX_LIMITS.items():
        _int(value[key], 1, maximum, key)
    return value


def _layers(value, maximum, label, allowed=None):
    if not isinstance(value, list) or len(value) > maximum + 1:
        raise ValueError(f"invalid {label}")
    for layer in value:
        _int(layer, 0, maximum, label)
    if len(set(value)) != len(value) or (allowed is not None and not set(value) <= set(allowed)):
        raise ValueError(f"unsupported or duplicate {label}")
    return value


def _returns(value):
    defaults = {"token_logprobs": False, "topk": 0, "hidden_states": [], "mlp_act": [], "attn_layers": []}
    if not isinstance(value, dict) or set(value) - set(defaults):
        raise ValueError("unknown teacher-forced return field")
    value = {**defaults, **value}
    if type(value["token_logprobs"]) is not bool:
        raise ValueError("token_logprobs must be boolean")
    _int(value["topk"], 0, 50, "topk")
    _layers(value["hidden_states"], 24, "hidden_states")
    _layers(value["mlp_act"], 23, "mlp_act")
    _layers(value["attn_layers"], 23, "attn_layers", FULL_ATTENTION_LAYERS)
    if not any(value.values()):
        raise ValueError("request at least one diagnostic observable")
    return value


def _chat(value):
    if not isinstance(value, list) or not value:
        raise ValueError("nonempty text-only chat required")
    for row in value:
        if (not isinstance(row, dict) or set(row) != {"role", "content"}
                or row["role"] not in ("system", "user", "assistant")
                or not isinstance(row["content"], str)):
            raise ValueError("unsupported chat message")


def _practice(items):
    from .benchmark_data import validate_item
    if not isinstance(items, list):
        raise ValueError("practice_items must be a list")
    result = {}
    for item in items:
        validate_item(item)
        if item["benchmark"] != "ifeval" or item["split"] == "test" or item["item_id"] in result:
            raise ValueError("unique non-test IFEval practice items required")
        result[item["item_id"]] = item
    return result


def validate_request(request, limits=None, practice_items=None):
    """Freeze effective controls without tokenization, loading, or generation."""
    limits = validate_limits(limits)
    practice = _practice(practice_items or [])
    if not isinstance(request, dict) or request.get("type") not in ("teacher_forced", "generate"):
        raise ValueError("probe type must be teacher_forced or generate")
    value = copy.deepcopy(request)
    if value.get("model") not in ("original", "edited", "both"):
        raise ValueError("probe model must be original, edited, or both")
    arms = 2 if value["model"] == "both" else 1
    if value["type"] == "teacher_forced":
        if set(value) - {"type", "model", "texts", "chats", "enable_thinking", "return"}:
            raise ValueError("unknown teacher-forced request field")
        if ("texts" in value) == ("chats" in value):
            raise ValueError("provide exactly one of texts or chats")
        source = value.get("texts", value.get("chats"))
        if not isinstance(source, list) or not 1 <= len(source) <= limits["max_texts"]:
            raise ValueError("teacher-forced text budget exceeded")
        if "texts" in value:
            if any(not isinstance(x, str) or not x.strip() for x in source):
                raise ValueError("nonempty raw texts required")
        else:
            for chat in source:
                _chat(chat)
        value.setdefault("enable_thinking", False)
        if type(value["enable_thinking"]) is not bool:
            raise ValueError("enable_thinking must be boolean")
        value["return"] = _returns(value.get("return"))
        if value["return"]["attn_layers"] and len(source) > limits["max_attention_texts"]:
            raise ValueError("attention text budget exceeded")
    else:
        if set(value) - {"type", "model", "problems", "prompts", "n", "max_tokens", "settings", "activations"}:
            raise ValueError("unknown generation request field")
        problems, prompts = value.setdefault("problems", []), value.setdefault("prompts", [])
        if (not isinstance(problems, list) or any(not isinstance(x, str) or x not in practice for x in problems)
                or len(set(problems)) != len(problems)):
            raise ValueError("generation problems must name unique available practice items")
        if not isinstance(prompts, list) or any(not isinstance(x, str) or not x.strip() for x in prompts):
            raise ValueError("custom prompts must be nonempty strings")
        value.setdefault("n", 1)
        _int(value["n"], 1, limits["max_generations"], "n")
        count = (len(problems) + len(prompts)) * value["n"] * arms
        if not 1 <= count <= limits["max_generations"]:
            raise ValueError("generation sequence budget exceeded (both arms count)")
        value.setdefault("max_tokens", min(1024, limits["max_new_tokens"]))
        _int(value["max_tokens"], 1, limits["max_new_tokens"], "max_tokens")
        raw = value.get("settings", {})
        if not isinstance(raw, dict) or set(raw) - set(DEFAULT_SETTINGS):
            raise ValueError("unknown generation settings")
        settings = value["settings"] = {**DEFAULT_SETTINGS, **raw}
        for key, lo, hi in (("temperature", 0, 10), ("top_p", 0, 1), ("presence_penalty", -2, 2)):
            if type(settings[key]) not in (int, float) or not math.isfinite(settings[key]) or not lo <= settings[key] <= hi:
                raise ValueError(f"invalid {key}")
        if settings["top_p"] == 0:
            raise ValueError("top_p must be positive")
        _int(settings["top_k"], 0, 248320, "top_k")
        _int(settings["seed"], 0, 2 ** 63 - 1, "seed")
        if type(settings["enable_thinking"]) is not bool:
            raise ValueError("enable_thinking must be boolean")
        activation = value.get("activations")
        if activation is not None:
            if not isinstance(activation, dict) or set(activation) - {"layers", "mlp", "max_answer_tokens"}:
                raise ValueError("invalid generation activation request")
            activation.setdefault("layers", [])
            activation.setdefault("mlp", False)
            activation.setdefault("max_answer_tokens", min(8192, value["max_tokens"]))
            _layers(activation["layers"], 24, "activation layers")
            if type(activation["mlp"]) is not bool or not activation["layers"]:
                raise ValueError("generation activations require layers and boolean mlp")
            _int(activation["max_answer_tokens"], 1, min(8192, value["max_tokens"]), "max_answer_tokens")
    if len(json.dumps(value, ensure_ascii=False).encode()) > limits["max_input_bytes"]:
        raise ValueError("probe input byte budget exceeded")
    return value


def paired_seed(base_seed, item_id, sample_index):
    """Same seed for both arms; independent of candidate, request order and hash."""
    return int.from_bytes(hashlib.sha256(f"{base_seed}\0{item_id}\0{sample_index}".encode()).digest()[:8], "big") & (2 ** 63 - 1)


def paired_summary(original, edited):
    key = lambda row: (row["item_id"], row["sample_index"], row["seed"])
    left, right = {key(row): row for row in original}, {key(row): row for row in edited}
    if len(left) != len(original) or len(right) != len(edited) or left.keys() != right.keys():
        raise ValueError("paired generation identities differ")
    deltas = []
    for identity, row in left.items():
        a, b = row.get("grade"), right[identity].get("grade")
        if a and b and a["status"] == b["status"] == "graded":
            deltas.append(int(b["correct"]) - int(a["correct"]))
    return {"paired_sequences": len(left), "graded_pairs": len(deltas),
            "ungraded_pairs": len(left) - len(deltas),
            "accuracy_delta": float(np.mean(deltas)) if deltas else None,
            "paired_standard_error": float(np.std(deltas, ddof=1) / math.sqrt(len(deltas))) if len(deltas) > 1 else None,
            "wrong_to_right": sum(x > 0 for x in deltas), "right_to_wrong": sum(x < 0 for x in deltas)}


def _write(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _seal(payload):
    return {"payload": payload, "sha256": canonical_hash(payload)}


class _Arrays:
    def __init__(self, directory, maximum):
        self.directory, self.maximum = Path(directory), maximum
        self.bytes, self.files = 0, {}

    def save(self, name, array):
        if Path(name).name != name or name in self.files:
            raise ValueError("invalid or duplicate array name")
        value = np.asarray(array)
        if value.dtype.hasobject or not np.all(np.isfinite(value)):
            raise ValueError("nonfinite or object diagnostic array")
        if self.bytes + value.nbytes > self.maximum:
            raise ValueError("diagnostic array byte budget exceeded")
        path = self.directory / name
        with path.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
        self.bytes += value.nbytes
        self.files[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": file_sha256(path)}


def _render_teacher(tokenizer, request, limits):
    texts = request.get("texts")
    if texts is None:
        texts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False,
                 enable_thinking=request["enable_thinking"]) for chat in request["chats"]]
    rows = []
    for index, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids or len(ids) > limits["max_input_tokens"] or len(ids) > limits["max_context_tokens"]:
            raise ValueError("teacher input token limit exceeded; truncation is forbidden")
        if request["return"]["attn_layers"] and len(ids) > limits["max_attention_tokens"]:
            raise ValueError("attention token limit exceeded")
        rows.append({"index": index, "text": text, "text_sha256": canonical_hash(text), "token_ids": ids})
    arms = 2 if request["model"] == "both" else 1
    if sum(len(row["token_ids"]) for row in rows) * arms > limits["max_total_tokens"]:
        raise ValueError("teacher total token budget exceeded (both arms count)")
    return rows


def _expected_array_bytes(rows, returns, config):
    width, mlp, heads = config.hidden_size, config.intermediate_size, config.num_attention_heads
    total = 0
    for row in rows:
        n = len(row["token_ids"])
        total += 4 * n * (len(returns["hidden_states"]) * width + len(returns["mlp_act"]) * mlp)
        total += 4 * n * n * heads * len(returns["attn_layers"])
        total += max(n - 1, 0) * ((4 if returns["token_logprobs"] else 0) + returns["topk"] * 12)
    return total


def _teacher_one(model, token_ids, returns, writer, prefix, torch):
    """Capture exact model internals; compute vocabulary probabilities in chunks."""
    backbone = model.model.language_model
    handles, observed = [], set()

    def save(key, tensor):
        if key in observed:
            raise ValueError("diagnostic module executed more than once")
        observed.add(key)
        array = tensor[0].detach().float().cpu().numpy().copy()
        writer.save(f"{prefix}-{key}.npy", array)

    try:
        for index in returns["hidden_states"]:
            module = backbone.embed_tokens if index == 0 else backbone.norm if index == 24 else backbone.layers[index - 1]
            handles.append(module.register_forward_hook(
                lambda _module, _args, output, index=index: save(f"hidden_{index}", output[0] if isinstance(output, tuple) else output)))
        for index in returns["mlp_act"]:
            handles.append(backbone.layers[index].mlp.down_proj.register_forward_pre_hook(
                lambda _module, inputs, index=index: save(f"mlp_{index}", inputs[0])))
        for index in returns["attn_layers"]:
            def attention(_module, _args, output, index=index):
                if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                    raise ValueError("requested full-attention weights unavailable in pinned runtime")
                save(f"attention_{index}", output[1])
            handles.append(backbone.layers[index].self_attn.register_forward_hook(attention))
        with torch.inference_mode():
            ids = torch.tensor([token_ids], dtype=torch.long, device="cpu")
            result = backbone(input_ids=ids, use_cache=False, output_attentions=bool(returns["attn_layers"]), return_dict=True)
            hidden = result.last_hidden_state
            logps, top_ids, top_logps = [], [], []
            if returns["token_logprobs"] or returns["topk"]:
                for start in range(0, len(token_ids) - 1, 32):
                    end = min(start + 32, len(token_ids) - 1)
                    logits = model.lm_head(hidden[:, start:end]).float()[0]
                    probabilities = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                    if returns["token_logprobs"]:
                        target = torch.tensor(token_ids[start + 1:end + 1], dtype=torch.long, device="cpu")
                        values = probabilities.gather(1, target[:, None])[:, 0]
                        logps.append(values.detach().cpu().numpy())
                    if returns["topk"]:
                        values, indices = torch.topk(probabilities, k=returns["topk"], dim=-1)
                        top_logps.append(values.detach().cpu().numpy())
                        top_ids.append(indices.detach().cpu().numpy())
            if returns["token_logprobs"]:
                writer.save(f"{prefix}-token_logprobs.npy", np.concatenate(logps) if logps else np.empty(0, dtype=np.float32))
            if returns["topk"]:
                shape = (0, returns["topk"])
                writer.save(f"{prefix}-topk_ids.npy", np.concatenate(top_ids) if top_ids else np.empty(shape, dtype=np.int64))
                writer.save(f"{prefix}-topk_logprobs.npy", np.concatenate(top_logps) if top_logps else np.empty(shape, dtype=np.float32))
        expected = {f"hidden_{i}" for i in returns["hidden_states"]} | {f"mlp_{i}" for i in returns["mlp_act"]} | {f"attention_{i}" for i in returns["attn_layers"]}
        if observed != expected:
            raise ValueError("requested diagnostic hooks did not all execute")
    finally:
        for handle in handles:
            handle.remove()


def _grade(row, item, source_sha, request, official_root):
    from .grading import grade_record
    policy = {"schema_version": 1, "reasoning_policy": "qwen_thinking" if request["settings"]["enable_thinking"] else "think_tags",
              "output_eos_tokens": ["<|im_end|>", "<|endoftext|>"], "truncation_policy": "grade",
              "generation_retry": {"mode": "none", "max_attempts": 1}, "grader_timeout_seconds": 20,
              "grader_retry": {"mode": "none", "max_attempts": 1}}
    outcome = {"status": "truncated" if row["truncated"] else "ok", "raw_output": row["raw_output"],
               "finish_reason": "length" if row["truncated"] else "eos", "token_counts": row["token_counts"],
               "duration_seconds": row["duration_seconds"], "error": None}
    record = {"schema_version": 1, "record_type": "run", "run_id": "research-diagnostic-" + canonical_hash(request)[:16],
              "evidence_kind": "real_model", "profile": "fresh_editor_search", "stage": "pilot",
              **{key: item[key] for key in ("benchmark", "dataset_revision", "item_id", "split")},
              "sample_index": row["sample_index"], "seed": row["seed"],
              "checkpoint": {"model_id": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17", "sha256": source_sha},
              "configuration_sha256": canonical_hash(request), "retry_policy": policy["generation_retry"],
              "attempts": [{"attempt_index": 0, **outcome}], **outcome}
    return grade_record(record, item, policy, ifeval_root=official_root)


def _generation_rows(tokenizer, request, practice, limits):
    items = [(identity, practice[identity]["task_text"]) for identity in request["problems"]]
    items += [(f"custom:{index}:" + canonical_hash(text)[:16], text) for index, text in enumerate(request["prompts"])]
    rows = []
    for identity, text in items:
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=request["settings"]["enable_thinking"])
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if (not ids or len(ids) > limits["max_input_tokens"]
                or len(ids) + request["max_tokens"] > limits["max_context_tokens"]):
            raise ValueError("generation context exceeds limit; truncation is forbidden")
        for sample in range(request["n"]):
            rows.append({"item_id": identity, "sample_index": sample,
                         "seed": paired_seed(request["settings"]["seed"], identity, sample),
                         "prompt": text, "rendered_prompt": rendered, "token_ids": ids})
    arms = 2 if request["model"] == "both" else 1
    # Reserve maximum generation work, including activation replay input tokens.
    reserved = sum(len(row["token_ids"]) + request["max_tokens"] for row in rows)
    if request.get("activations"):
        reserved += sum(len(row["token_ids"]) + request["activations"]["max_answer_tokens"] for row in rows)
    if reserved * arms > limits["max_generation_tokens"]:
        raise ValueError("generation/replay total token reservation exceeds budget")
    return rows


def _eos_ids(*values):
    result = set()
    for value in values:
        if value is None:
            continue
        for token in value if isinstance(value, (list, tuple)) else [value]:
            if type(token) is not int or token < 0:
                raise ValueError("explicit nonnegative source EOS token IDs required")
            result.add(token)
    if not result:
        raise ValueError("explicit source EOS token IDs required")
    return sorted(result)


def _generate_one(model, tokenizer, row, request, torch):
    from .native_backend import _PresencePenalty
    from transformers import GenerationConfig, LogitsProcessorList
    ids = row["token_ids"]
    eos = _eos_ids(tokenizer.eos_token_id, model.config.text_config.eos_token_id)
    settings = request["settings"]
    config = GenerationConfig(max_new_tokens=request["max_tokens"], do_sample=settings["temperature"] > 0,
        temperature=settings["temperature"] if settings["temperature"] > 0 else 1.0,
        top_p=settings["top_p"], top_k=settings["top_k"], repetition_penalty=1.0,
        eos_token_id=eos, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0],
        bos_token_id=None, use_cache=True, num_beams=1, num_return_sequences=1)
    torch.manual_seed(row["seed"])
    started = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(input_ids=torch.tensor([ids], dtype=torch.long, device="cpu"),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device="cpu"), generation_config=config,
            logits_processor=LogitsProcessorList([_PresencePenalty(len(ids), settings["presence_penalty"])]),
            logits_to_keep=1)
    tokens = generated[0, len(ids):].tolist()
    if not tokens or (tokens[-1] not in eos and len(tokens) != request["max_tokens"]):
        raise ValueError("generation terminated unexpectedly")
    return {**row, "output_token_ids": tokens,
            "raw_output": tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "truncated": tokens[-1] not in eos, "duration_seconds": time.monotonic() - started,
            "token_counts": {"input": len(ids), "output": len(tokens), "total": len(ids) + len(tokens)}}


def _run_worker(payload, directory):
    from .native_backend import TRANSFORMERS_VERSIONS, _check_versions, _preserve_and_verify_loaded_tensors
    _check_versions(TRANSFORMERS_VERSIONS)
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    directory = Path(directory)
    limits, request = payload["limits"], payload["request"]
    torch.set_num_threads(limits["cpu_threads"])
    practice = _practice(payload["practice_items"])
    tokenizer = AutoTokenizer.from_pretrained(payload["original"], local_files_only=True, trust_remote_code=False)
    inputs = (_render_teacher(tokenizer, request, limits) if request["type"] == "teacher_forced"
              else _generation_rows(tokenizer, request, practice, limits))
    arms = ["original", "edited"] if request["model"] == "both" else [request["model"]]
    progress = {"model_loads_started": 0, "model_loads_completed": 0,
                "model_forward_calls_started": 0, "model_forward_calls_completed": 0, "forward_sequences_started": 0,
                "forward_sequences_completed": 0, "generation_sequences_started": 0, "generation_sequences_completed": 0,
                "teacher_forced_sequences_started": 0, "teacher_forced_sequences_completed": 0,
                "activation_replay_sequences_started": 0, "activation_replay_sequences_completed": 0,
                "input_tokens": 0, "generated_tokens": 0, "replay_tokens": 0}
    def mark(kind, count=1):
        progress[kind] += count
        _write(directory / "progress.json", progress)
    arrays = _Arrays(directory, limits["max_array_bytes"])
    generations, loading = {}, {}
    began = time.monotonic()
    for arm in arms:
        path = Path(payload[arm])
        expected = payload[arm + "_sha256"]
        if canonical_hash(inventory_model(path)["files"]) != expected:
            raise ValueError("probe checkpoint identity changed before loading")
        mark("model_loads_started")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(str(path), local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="eager")
        model.eval()
        model.requires_grad_(False)
        loading[arm] = _preserve_and_verify_loaded_tensors(model, path)
        mark("model_loads_completed")
        forward_handles = [model.model.language_model.register_forward_pre_hook(
            lambda _module, _inputs: mark("model_forward_calls_started")),
            model.model.language_model.register_forward_hook(
                lambda _module, _inputs, _output: mark("model_forward_calls_completed"))]
        returns = request.get("return")
        if returns and _expected_array_bytes(inputs, returns, model.config.text_config) * len(arms) > limits["max_array_bytes"]:
            raise ValueError("requested arrays exceed budget before forward")
        rows = []
        for index, row in enumerate(inputs):
            mark("forward_sequences_started")
            mark("input_tokens", len(row["token_ids"]))
            if request["type"] == "teacher_forced":
                mark("teacher_forced_sequences_started")
                _teacher_one(model, row["token_ids"], returns, arrays, f"{arm}-{index:04d}", torch)
                mark("teacher_forced_sequences_completed")
                rows.append(row)
            else:
                mark("generation_sequences_started")
                result = _generate_one(model, tokenizer, row, request, torch)
                mark("generation_sequences_completed")
                mark("generated_tokens", len(result["output_token_ids"]))
                # A grading failure must never erase an already generated answer.
                result["grade"] = None
                rows.append(result)
                _write(directory / f"{arm}-generations.json", rows)
                result["grade"] = (_grade(result, practice[row["item_id"]], expected, request,
                    payload["official_ifeval_root"]) if row["item_id"] in practice else None)
                # Persist each completed result before optional replay or next sample.
                _write(directory / f"{arm}-generations.json", rows)
                if request.get("activations"):
                    activation = request["activations"]
                    replay = row["token_ids"] + result["output_token_ids"][:activation["max_answer_tokens"]]
                    ret = {"token_logprobs": False, "topk": 0, "hidden_states": activation["layers"],
                           "mlp_act": [i for i in activation["layers"] if i < 24] if activation["mlp"] else [], "attn_layers": []}
                    if arrays.bytes + _expected_array_bytes([{"token_ids": replay}], ret, model.config.text_config) > limits["max_array_bytes"]:
                        raise ValueError("generation activation replay arrays exceed budget")
                    mark("activation_replay_sequences_started")
                    mark("forward_sequences_started")
                    mark("replay_tokens", len(replay))
                    _teacher_one(model, replay, ret, arrays, f"{arm}-{index:04d}-replay", torch)
                    mark("activation_replay_sequences_completed")
                    mark("forward_sequences_completed")
                    result["activation_replay"] = {"token_ids": replay, "prompt_tokens": len(row["token_ids"]),
                        "answer_tokens": len(replay) - len(row["token_ids"]), "arrays_prefix": f"{arm}-{index:04d}-replay"}
                    _write(directory / f"{arm}-generations.json", rows)
            mark("forward_sequences_completed")
        _write(directory / f"{arm}-{'texts' if request['type'] == 'teacher_forced' else 'generations'}.json", rows)
        generations[arm] = rows
        if canonical_hash(inventory_model(path)["files"]) != expected:
            raise ValueError("probe checkpoint changed during forward")
        for handle in forward_handles:
            handle.remove()
        del model
        gc.collect()
    files = {p.name: {"sha256": file_sha256(p), "size_bytes": p.stat().st_size}
             for p in sorted(directory.iterdir()) if p.suffix == ".npy" or p.name.endswith(("-generations.json", "-texts.json"))}
    summaries = {}
    if request["type"] == "generate":
        for arm, rows in generations.items():
            grades = [r["grade"] for r in rows if r["grade"] and r["grade"]["status"] == "graded"]
            summaries[arm] = {"sequences": len(rows), "graded": len(grades), "ungraded": len(rows) - len(grades),
                "accuracy": sum(g["correct"] for g in grades) / len(grades) if grades else None,
                "mean_output_tokens": float(np.mean([r["token_counts"]["output"] for r in rows])),
                "truncated": sum(r["truncated"] for r in rows)}
        if request["model"] == "both":
            summaries["paired"] = paired_summary(generations["original"], generations["edited"])
    result = {"schema_version": 1, "kind": "research_diagnostic_probe", "status": "completed",
              "request_sha256": canonical_hash(request), "original_sha256": payload["original_sha256"],
              "edited_sha256": payload.get("edited_sha256"), "practice_items_sha256": canonical_hash(payload["practice_items"]),
              "backend": "transformers_cpu_bfloat16", "versions": TRANSFORMERS_VERSIONS,
              "browser_activations": False, "final_comparison": False, "training": False,
              "accounting": progress, "arrays": arrays.files, "array_bytes": arrays.bytes,
              "token_probability_alignment": "index i predicts input token_ids[i+1]; first token unscored",
              "hidden_state_indexing": "0 embedding; 1..23 post-block; 24 final normalized decoder state",
              "mlp_indexing": "MLP i is the down_proj input of zero-based block i; hidden i+1 is its block output except final norm at 24",
              "attention_scope": "full-attention probabilities, heads x queries x keys; no linear-attention surrogate",
              "generation_activation_method": "separately counted teacher-forced replay of prompt plus bounded answer prefix",
              "files": files, "model_loading": loading, "summaries": summaries,
              "elapsed_seconds": time.monotonic() - began}
    _write(directory / "manifest.json", _seal(result))


def run_probe(original, candidate_model, request, output, limits=None, practice_items=None, *,
              expected_original_sha256=None, expected_candidate_sha256=None, official_ifeval_root=None,
              python_executable=None):
    """Execute exactly one supervised tool attempt, retaining all failure artifacts.

    The study reserves the tool attempt before calling this function. An edited
    model must already be materialized and immutable. This function checks both
    source identities, all non-weight assets, and hashes emitted artifacts; it
    does not authorize edits. Limits are sampled RSS and wall bounds, not a claim
    of OS containment. No automatic retries or external downloads occur.
    """
    from .mechanistic_tools import _rss_bytes
    from .editor_sandbox import _WATCHDOG
    limits = validate_limits(limits)
    request = validate_request(request, limits, practice_items)
    original, output = Path(original).resolve(), Path(output).resolve()
    candidate = Path(candidate_model).resolve() if candidate_model is not None else None
    if request["model"] != "original" and candidate is None:
        raise ValueError("edited probe requires an immutable candidate model")
    paths = [original] + ([candidate] if candidate else [])
    if output.exists() or any(output == p or p in output.parents or output in p.parents for p in paths):
        raise ValueError("probe output must be new and outside model directories")
    before = inventory_model(original)
    identities = {"original": canonical_hash(before["files"])}
    inventories = {"original": before}
    if candidate:
        after = inventory_model(candidate)
        inventories["edited"] = after
        identities["edited"] = canonical_hash(after["files"])
        # Tokenizer/config/templates and complete tensor shape/dtype contract stay fixed.
        shape = lambda inv: {k: (v["shape"], v["dtype"]) for k, v in inv["tensors"].items()}
        nonweights = lambda inv: {k: v for k, v in inv["files"].items() if not k.endswith(".safetensors")}
        if shape(before) != shape(after) or nonweights(before) != nonweights(after):
            raise ValueError("edited probe changed model controls or tensor contract")
    for arm, expected in (("original", expected_original_sha256), ("edited", expected_candidate_sha256)):
        if expected is not None and identities.get(arm) != expected:
            raise ValueError("probe source identity differs from expected snapshot")
    if any(sum(row["size_bytes"] for row in inv["files"].values()) > limits["max_checkpoint_bytes"] for inv in inventories.values()):
        raise ValueError("checkpoint byte budget exceeded")
    if request["type"] == "generate" and request["problems"]:
        from .grading import verify_ifeval_assets
        if official_ifeval_root is None:
            raise ValueError("official IFEval assets required for practice grading")
        verify_ifeval_assets(official_ifeval_root)
    output.mkdir(parents=True, mode=0o700)
    payload = {"original": str(original), "edited": str(candidate) if candidate else None,
               "original_sha256": identities["original"], "edited_sha256": identities.get("edited"),
               "request": request, "limits": limits, "practice_items": practice_items or [],
               "official_ifeval_root": str(Path(official_ifeval_root).resolve()) if official_ifeval_root else None}
    _write(output / "request.json", payload)
    process, peak, success = None, 0, False
    reader = writer = -1
    worker_pid, missing_rss, rss_samples = None, 0, 0
    cleanup_error = None
    started = time.monotonic()
    try:
        # Preserve a virtualenv launcher symlink: resolving it loses its environment.
        executable = os.path.abspath(python_executable or sys.executable)
        command = [executable, "-I", "-B", "-c", _WORKER_BOOTSTRAP,
                   str(Path(__file__).resolve().parents[1]), str(output)]
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                       "CUDA_VISIBLE_DEVICES": "", "HF_DATASETS_OFFLINE": "1"}
        reader, writer = os.pipe()
        with (output / "worker.stdout").open("xb") as stdout, (output / "worker.stderr").open("xb") as stderr:
            process = subprocess.Popen([executable, "-I", "-S", "-B", "-c", _WATCHDOG,
                str(reader), str(started + limits["wall_seconds"]), json.dumps(command)],
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=environment,
                cwd=Path(__file__).resolve().parents[1], start_new_session=True,
                pass_fds=(reader,), close_fds=True)
            os.close(reader)
            reader = -1
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > limits["wall_seconds"]:
                    raise TimeoutError("diagnostic probe wall-time budget exhausted")
                if worker_pid is None:
                    identity_path = output / "worker-identity.json"
                    if identity_path.is_file():
                        identity = read_json(identity_path)
                        if (type(identity.get("pid")) is not int or identity["pid"] <= 0
                                or identity.get("parent_pid") != process.pid
                                or identity.get("process_group") != identity["pid"]):
                            raise RuntimeError("diagnostic worker process identity mismatch")
                        worker_pid = identity["pid"]
                    elif elapsed > 10:
                        raise RuntimeError("diagnostic worker failed to identify its process")
                if worker_pid is not None:
                    observed = _rss_bytes(worker_pid)
                    rss_samples += int(observed is not None)
                    missing_rss = missing_rss + 1 if observed is None else 0
                    # A completed child can vanish just before its guardian's
                    # next poll/reap. A brief grace avoids misclassifying that
                    # normal shutdown as unavailable memory telemetry.
                    if missing_rss > 2 and process.poll() is None:
                        raise RuntimeError("diagnostic worker RSS measurement unavailable")
                    peak = max(peak, observed or 0)
                    if peak > limits["max_rss_bytes"]:
                        raise MemoryError("diagnostic probe sampled RSS budget exhausted")
                if sum((output / name).stat().st_size for name in ("worker.stdout", "worker.stderr")) > limits["max_log_bytes"]:
                    raise RuntimeError("diagnostic worker output budget exhausted")
                time.sleep(0.1)
        if process.returncode != 0:
            if process.returncode == 124:
                raise TimeoutError("diagnostic probe watchdog wall-time budget exhausted")
            raise RuntimeError("diagnostic worker failed; retained worker.stderr and progress.json contain evidence")
        manifest = read_json(output / "manifest.json")
        value = manifest["payload"]
        if (manifest["sha256"] != canonical_hash(value) or value["request_sha256"] != canonical_hash(request)
                or value["original_sha256"] != identities["original"] or value["edited_sha256"] != identities.get("edited")):
            raise ValueError("probe manifest identity differs")
        for name, descriptor in value["files"].items():
            if Path(name).name != name or file_sha256(output / name) != descriptor["sha256"] or (output / name).stat().st_size != descriptor["size_bytes"]:
                raise ValueError("probe artifact identity differs")
        for arm, path in (("original", original), ("edited", candidate)):
            if path is not None and inventory_model(path) != inventories[arm]:
                raise ValueError("checkpoint changed during probe")
        success = True
        return manifest
    except BaseException as error:
        _write(output / "failure.json", _seal({"status": "failed", "error_type": type(error).__name__,
            "message": str(error), "request_sha256": canonical_hash(request), "automatic_retry": False}))
        raise
    finally:
        for descriptor in (reader, writer):
            if descriptor >= 0:
                os.close(descriptor)
        if process is not None and process.poll() is None:
            # Closing the sole parent-held writer triggers independent group
            # termination, including when this controller is SIGKILLed. Never
            # kill the guardian first: it owns worker termination and reaping.
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cleanup_error = "probe watchdog did not confirm worker termination"
                success = False
                _write(output / "failure.json", _seal({"status": "failed", "error_type": "RuntimeError",
                    "message": cleanup_error, "request_sha256": canonical_hash(request), "automatic_retry": False}))
        _write(output / "supervision.json", _seal({"status": "completed" if success else "failed",
            "elapsed_seconds": time.monotonic() - started, "limits": limits,
            "sampled_worker_peak_rss_bytes": peak if rss_samples else None, "rss_samples": rss_samples,
            "process_exit_confirmed": process is None or process.poll() is not None,
            "worker_pid": worker_pid, "watchdog_pid": process.pid if process is not None else None,
            "watchdog": "independent process; parent pipe EOF or wall deadline terminates worker process group",
            "cleanup_error": cleanup_error,
            "python_executable": os.path.abspath(python_executable or sys.executable),
            "memory_scope": "sampled CPU worker RSS; excludes grader subprocesses and WebGPU",
            "os_isolation": "unverified", "automatic_retries": 0}))
        if cleanup_error:
            raise RuntimeError(cleanup_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    args = parser.parse_args()
    _run_worker(read_json(Path(args.worker) / "request.json"), Path(args.worker))


if __name__ == "__main__":
    main()
