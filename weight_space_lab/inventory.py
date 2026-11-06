"""Byte-level inventories of complete local Safetensors model snapshots.

No model code is imported or executed. Stored tensor dtypes and shapes come
from the serialized headers; parameter counts include every stored tensor,
including buffers and any separately stored tied weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path

from .replay import _header, _object


_CHUNK_BYTES = 8 * 1024 * 1024


def safe_relative(value):
    """Validate a portable, unambiguous relative artifact path."""
    if (not isinstance(value, str) or not value or "\\" in value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or any(part in ("", ".", "..") for part in value.split("/"))
            or ":" in value):
        raise ValueError("artifact path must be a normalized relative path")
    return value


def read_json(path):
    """Read strict JSON, rejecting duplicate keys and nonfinite constants."""
    def reject_constant(_):
        raise ValueError("JSON must not contain nonfinite constants")

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_object, parse_constant=reject_constant)


def _hash_stream(stream, length=None):
    digest = hashlib.sha256()
    remaining = length
    while remaining is None or remaining:
        chunk = stream.read(_CHUNK_BYTES if remaining is None
                            else min(_CHUNK_BYTES, remaining))
        if not chunk:
            if remaining:
                raise ValueError("artifact ended before the declared byte span")
            break
        digest.update(chunk)
        if remaining is not None:
            remaining -= len(chunk)
    return digest.hexdigest()


def file_sha256(path):
    """Hash a regular file without loading it into memory."""
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("artifact must be a regular file, not a link or device")
    with path.open("rb") as stream:
        return _hash_stream(stream)


def _files(root):
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("model snapshot must be a directory, not a link")
    files = {}
    directories = [root]
    while directories:
        directory = directories.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = safe_relative(path.relative_to(root).as_posix())
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISDIR(mode):
                    directories.append(path)
                elif stat.S_ISREG(mode):
                    files[relative] = path
                else:
                    raise ValueError("snapshot contains a link or nonregular file")
    return dict(sorted(files.items()))


def _chat_templates(files, tokenizer_config):
    templates = {}
    for name, path in files.items():
        if name.endswith(".jinja"):
            content = path.read_text(encoding="utf-8")
            if not content.strip():
                raise ValueError("chat template must not be empty")
            templates[name] = {"sha256": file_sha256(path), "content": content}
    embedded = tokenizer_config.get("chat_template")
    if embedded is not None:
        if isinstance(embedded, str):
            embedded = {"default": embedded}
        elif isinstance(embedded, list):
            converted = {}
            for entry in embedded:
                if (not isinstance(entry, dict) or set(entry) != {"name", "template"}
                        or not isinstance(entry["name"], str) or not entry["name"]
                        or entry["name"] in converted):
                    raise ValueError("invalid named chat template")
                converted[entry["name"]] = entry["template"]
            embedded = converted
        if not isinstance(embedded, dict) or not embedded:
            raise ValueError("invalid embedded chat template")
        for name, content in embedded.items():
            if (not isinstance(name, str) or not name or not isinstance(content, str)
                    or not content.strip()):
                raise ValueError("invalid embedded chat template")
            templates[f"tokenizer_config.json#chat_template/{name}"] = {
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "content": content,
            }
    if not templates:
        raise ValueError("snapshot has no chat template")
    return dict(sorted(templates.items()))


def inventory_model(path):
    """Inspect every file and stored tensor in a complete model directory.

    Sharded checkpoints must have an exact index: missing, duplicate, extra,
    and incorrectly assigned tensors are rejected. Single-file checkpoints
    without an index must use the canonical ``model.safetensors`` filename.
    Inputs must remain stable during inspection. The result contains only
    paths relative to the snapshot, never the machine's snapshot location.
    """
    root = Path(path)
    files = _files(root)
    for required in ("config.json", "tokenizer_config.json"):
        if required not in files:
            raise ValueError(f"snapshot is missing required asset: {required}")
    if not ("tokenizer.json" in files or "tokenizer.model" in files
            or {"vocab.json", "merges.txt"} <= files.keys()):
        raise ValueError("snapshot is missing tokenizer data")
    model_config = read_json(files["config.json"])
    tokenizer_config = read_json(files["tokenizer_config.json"])
    if not isinstance(model_config, dict) or not isinstance(tokenizer_config, dict):
        raise ValueError("model and tokenizer configurations must be JSON objects")
    chat_templates = _chat_templates(files, tokenizer_config)
    file_inventory = {name: {"sha256": file_sha256(file),
                             "size_bytes": file.stat().st_size}
                      for name, file in files.items()}
    shards = sorted(name for name in files if name.endswith(".safetensors"))
    if not shards:
        raise ValueError("snapshot contains no Safetensors weights")
    tensors, shard_layout = {}, {}
    stored_bytes = 0
    for name in shards:
        with files[name].open("rb") as stream:
            header, data_start = _header(stream)
            shard_layout[name] = {"data_start": data_start,
                                  "metadata": header.get("__metadata__", {})}
            for tensor_name, tensor in header.items():
                if tensor_name == "__metadata__":
                    continue
                if not tensor_name or tensor_name in tensors:
                    raise ValueError("snapshot tensor names must be nonempty and unique")
                start, end = tensor["data_offsets"]
                stream.seek(data_start + start)
                tensors[tensor_name] = {
                    "shard": name,
                    "shape": tensor["shape"],
                    "dtype": tensor["dtype"],
                    "parameter_count": math.prod(tensor["shape"]),
                    "data_offsets": tensor["data_offsets"],
                    "data_start": data_start,
                    "sha256": _hash_stream(stream, end - start),
                }
                stored_bytes += end - start
    if not tensors:
        raise ValueError("snapshot contains no stored tensors")
    index = None
    index_path = "model.safetensors.index.json"
    if index_path in files:
        content = read_json(files[index_path])
        if (not isinstance(content, dict) or not isinstance(content.get("weight_map"), dict)
                or not content["weight_map"]):
            raise ValueError("Safetensors index must contain a nonempty weight_map")
        weight_map = content["weight_map"]
        for shard in weight_map.values():
            safe_relative(shard)
        actual_map = {name: tensor["shard"] for name, tensor in tensors.items()}
        if weight_map != actual_map or set(weight_map.values()) != set(shards):
            raise ValueError("Safetensors index does not match the stored tensor layout")
        metadata = content.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Safetensors index metadata must be an object")
        if "total_size" in metadata and (type(metadata["total_size"]) is not int
                                         or metadata["total_size"] != stored_bytes):
            raise ValueError("Safetensors index total_size does not match stored tensor bytes")
        index = {"path": index_path, "content": content}
    elif shards != ["model.safetensors"]:
        raise ValueError("sharded or noncanonical weights require a Safetensors index")
    tokenizer_names = {"tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                       "vocab.json", "merges.txt", "added_tokens.json",
                       "special_tokens_map.json", "preprocessor_config.json",
                       "processor_config.json", "video_preprocessor_config.json"}
    return {
        "schema_version": 1,
        "files": file_inventory,
        "tensors": dict(sorted(tensors.items())),
        "shards": shards,
        "shard_layout": shard_layout,
        "index": index,
        "parameter_count": sum(tensor["parameter_count"] for tensor in tensors.values()),
        "parameter_count_scope": "all stored tensors, including buffers and stored aliases",
        "stored_tensor_bytes": stored_bytes,
        "model_config": model_config,
        "tokenizer_assets": [name for name in files if name in tokenizer_names
                             or name.endswith(".jinja")],
        "chat_templates": chat_templates,
    }
