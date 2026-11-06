"""Bounded, pinned acquisition of model files and the selected published patch.

Only public HTTPS endpoints are used. Credentials, redirect URLs, repository
history, and unrelated upstream files are neither requested nor recorded.
The completed bundle is a read-only snapshot; its audit is finalized with it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .inventory import file_sha256, inventory_model, read_json, safe_relative


DEFAULT_MAX_DOWNLOAD_BYTES = 8 * 1024**3
_METADATA_LIMIT = 16 * 1024**2
_ASSETS = {
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer.model",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "vocab.json", "merges.txt", "chat_template.jinja", "chat_template.json",
    "processor_config.json", "preprocessor_config.json",
    "video_preprocessor_config.json", "image_processor_config.json",
}


def validate_reference(reference):
    """Validate immutable pins without promoting expectations to observations."""
    if (not isinstance(reference, dict) or type(reference.get("schema_version")) is not int
            or reference["schema_version"] != 1):
        raise ValueError("unsupported published reference")
    if reference.get("claim_type") != "published_patch_replay":
        raise ValueError("reference must describe a published patch replay")
    for field in ("source_revision", "base_revision"):
        if not isinstance(reference.get(field), str) or not re.fullmatch(r"[0-9a-f]{40}", reference[field]):
            raise ValueError("source and model revisions must be immutable 40-digit identifiers")
    if not isinstance(reference.get("source_repository"), str) or not re.fullmatch(
            r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", reference["source_repository"]):
        raise ValueError("source_repository must be a public GitHub repository URL")
    if not isinstance(reference.get("base_model"), str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", reference["base_model"]):
        raise ValueError("base_model must be an owner/model identifier")
    for field in ("manifest_path", "archive_path"):
        safe_relative(reference.get(field))
    if reference["manifest_path"] == reference["archive_path"]:
        raise ValueError("manifest and archive must be distinct files")
    names = reference.get("tensor_names")
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("reference tensor_names must be unique nonempty strings")
    for field in ("expected_changed_stored_entries", "reported_targeted_entries"):
        if type(reference.get(field)) is not int or reference[field] < 0:
            raise ValueError("reference counts must be nonnegative integers")
    return reference


class _HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise ValueError("non-HTTPS redirect refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _HTTP:
    """No retries or credentials; finite byte and checked elapsed-time budgets.

    A public urllib response does not expose a portable per-read timeout setter.
    Each request starts with a socket timeout bounded by its remaining budget.
    Incremental read1 calls avoid waiting for a whole buffer under slow traffic.
    Deadlines are checked before and after every read; an in-progress socket
    operation can delay failure until control returns. This is not a hard
    process-interruption deadline.
    """

    def __init__(self, max_bytes, timeout, max_elapsed_seconds=3600.0):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_download_bytes must be a positive integer")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        self.remaining = max_bytes
        self.timeout = float(timeout)
        if (not isinstance(max_elapsed_seconds, (int, float)) or isinstance(max_elapsed_seconds, bool)
                or not math.isfinite(max_elapsed_seconds) or max_elapsed_seconds <= 0):
            raise ValueError("max_elapsed_seconds must be a positive finite number")
        self.deadline = time.monotonic() + max_elapsed_seconds
        self.opener = urllib.request.build_opener(_HTTPSRedirects())

    def download(self, url, destination, *, size=None, limit=None):
        if urllib.parse.urlparse(url).scheme != "https":
            raise ValueError("only HTTPS downloads are supported")
        cap = min(self.remaining, limit if limit is not None else self.remaining)
        if size is not None and size > cap:
            raise ValueError("download exceeds the remaining byte budget")
        started = time.monotonic()
        request_timeout = min(self.timeout, self.deadline - started)
        if request_timeout <= 0:
            raise TimeoutError("acquisition exceeded its elapsed-time budget")
        request = urllib.request.Request(url, headers={"User-Agent": "weight-space-lab/0.1", "Accept": "application/json"})
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            with destination.open("xb") as target:
                created = True
                source = self.opener.open(request, timeout=request_timeout)
                with source:
                    self._transfer(source, target, started, request_timeout, cap, size)
        except BaseException:
            if created:
                destination.unlink(missing_ok=True)
            raise

    def _transfer(self, source, target, started, request_timeout, cap, size):
        if urllib.parse.urlparse(source.geturl()).scheme != "https":
            raise ValueError("non-HTTPS response refused")
        total = 0
        read = getattr(source, "read1", source.read)
        while True:
            if time.monotonic() - started > request_timeout:
                raise TimeoutError("download exceeded its elapsed-time budget")
            chunk = read(min(1024**2, cap - total + 1))
            if time.monotonic() - started > request_timeout:
                raise TimeoutError("download exceeded its elapsed-time budget")
            if not chunk:
                break
            total += len(chunk)
            self.remaining -= len(chunk)
            if total > cap:
                raise ValueError("download exceeds the byte budget")
            target.write(chunk)
        if size is not None and total != size:
            raise ValueError("download size does not match remote metadata")
        target.flush()
        os.fsync(target.fileno())

    def metadata(self, url, directory):
        path = Path(directory) / ".metadata.json"
        self.download(url, path, limit=_METADATA_LIMIT)
        try:
            return read_json(path)
        finally:
            path.unlink(missing_ok=True)


def _identity(metadata, *, github=False):
    size = metadata.get("size")
    if type(size) is not int or size < 0:
        raise ValueError("remote file metadata has no valid size")
    lfs = metadata.get("lfs")
    if lfs is not None:
        digest = lfs.get("sha256") if isinstance(lfs, dict) else None
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or lfs.get("size") != size:
            raise ValueError("invalid LFS file identity")
        return {"algorithm": "sha256", "digest": digest, "size_bytes": size}
    digest = metadata.get("sha" if github else "blobId")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{40}", digest):
        raise ValueError("remote file metadata has no immutable blob identity")
    return {"algorithm": "git_sha1", "digest": digest, "size_bytes": size}


def _check_identity(path, identity):
    if not isinstance(identity, dict) or set(identity) != {"algorithm", "digest", "size_bytes", "source_path", "source"}:
        raise ValueError("invalid recorded remote identity")
    safe_relative(identity["source_path"])
    size = identity["size_bytes"]
    if type(size) is not int or size < 0 or Path(path).stat().st_size != size:
        raise ValueError("file size disagrees with pinned source metadata")
    algorithm = identity["algorithm"]
    if algorithm == "sha256":
        observed = file_sha256(path)
    elif algorithm == "git_sha1":
        digest = hashlib.sha1(f"blob {size}\0".encode())
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
        observed = digest.hexdigest()
    else:
        raise ValueError("unsupported remote identity algorithm")
    if observed != identity["digest"]:
        raise ValueError("downloaded file hash disagrees with pinned source metadata")


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _payload_files(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("acquisition must be a real directory")
    files = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError("bundle must contain only regular files and directories")
        if path.is_file() and path.relative_to(root).as_posix() != "acquisition.json":
            name = path.relative_to(root).as_posix()
            files[name] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    return files


def _freeze(root):
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        path.chmod(0o555)
    root.chmod(0o555)


def _remove_stage(root):
    if root.exists():
        root.chmod(0o755)
        for path in root.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
        shutil.rmtree(root)


def _asset(name):
    return (name in _ASSETS
            or bool(re.fullmatch(r"chat_templates/[A-Za-z0-9_.-]+\.jinja", name))
            or bool(re.fullmatch(r"(?:LICENSE|NOTICE|COPYING)(?:[._-][A-Za-z0-9_.-]+)?", name, re.I)))


def acquire_published(config_path, output, *, max_download_bytes=DEFAULT_MAX_DOWNLOAD_BYTES,
                      timeout=60.0, max_elapsed_seconds=3600.0):
    """Acquire exactly the pinned snapshot and two selected patch files.

    Timeout applies to checked elapsed time for each complete HTTP transfer;
    max_elapsed_seconds applies across requests. An in-progress socket operation
    can delay reporting a deadline until control returns; this is not a hard
    process-interruption deadline. Byte budget includes metadata. No download
    is silently retried. Existing outputs are refused; incomplete staging
    directories never become bundles.
    """
    reference = validate_reference(read_json(config_path))
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("acquisition output must be a new path")
    http = _HTTP(max_download_bytes, timeout, max_elapsed_seconds)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".acquisition-incomplete-", dir=output.parent))
    try:
        repository = reference["source_repository"].removeprefix("https://github.com/")
        source_revision = reference["source_revision"]
        commit = http.metadata(f"https://api.github.com/repos/{repository}/git/commits/{source_revision}", stage)
        if not isinstance(commit, dict) or commit.get("sha") != source_revision:
            raise ValueError("resolved patch source revision differs from the reference")
        remote_files = {}
        for field, target_name in (("manifest_path", "manifest.json"), ("archive_path", "edit.npz")):
            source_path = reference[field]
            escaped = urllib.parse.quote(source_path, safe="/")
            meta = http.metadata(f"https://api.github.com/repos/{repository}/contents/{escaped}?ref={source_revision}", stage)
            if not isinstance(meta, dict) or meta.get("type") != "file" or meta.get("path") != source_path:
                raise ValueError("selected patch path did not resolve to a regular file")
            identity = {**_identity(meta, github=True), "source": "published_patch", "source_path": source_path}
            local = f"patch/{target_name}"
            http.download(f"https://raw.githubusercontent.com/{repository}/{source_revision}/{escaped}",
                          stage / local, size=identity["size_bytes"])
            _check_identity(stage / local, identity)
            remote_files[local] = identity
        model = reference["base_model"]
        revision = reference["base_revision"]
        model_meta = http.metadata(f"https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true", stage)
        if not isinstance(model_meta, dict) or model_meta.get("sha") != revision:
            raise ValueError("resolved model revision differs from the reference")
        siblings = model_meta.get("siblings")
        if not isinstance(siblings, list):
            raise ValueError("model metadata has no file inventory")
        available = {}
        for item in siblings:
            if not isinstance(item, dict):
                raise ValueError("invalid model file metadata")
            name = safe_relative(item.get("rfilename"))
            if name in available:
                raise ValueError("duplicate model file metadata")
            available[name] = item

        def download_model(name):
            if name not in available:
                raise ValueError("required model asset is missing from pinned revision")
            identity = {**_identity(available[name]), "source": "base_model", "source_path": name}
            local = f"model/{name}"
            http.download(f"https://huggingface.co/{model}/resolve/{revision}/{urllib.parse.quote(name, safe='/')}",
                          stage / local, size=identity["size_bytes"])
            _check_identity(stage / local, identity)
            remote_files[local] = identity

        index = "model.safetensors.index.json"
        if index in available:
            download_model(index)
            index_value = read_json(stage / "model" / index)
            weight_map = index_value.get("weight_map") if isinstance(index_value, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("model shard index has no weight_map")
            shards = {safe_relative(name) for name in weight_map.values()}
            if any(not name.endswith(".safetensors") for name in shards):
                raise ValueError("shard index must select safetensors files")
        elif "model.safetensors" in available:
            shards = {"model.safetensors"}
        else:
            raise ValueError("pinned model has no supported safetensors layout")
        selected = shards | {name for name in available if _asset(name)}
        if sum(_identity(available[name])["size_bytes"] for name in selected if name in available) > http.remaining:
            raise ValueError("selected model files exceed the remaining byte budget")
        for name in sorted(selected):
            download_model(name)
        _write_json(stage / "reference.json", reference)
        inventory = inventory_model(stage / "model")
        audit = {"schema_version": 1, "kind": "pinned_acquisition", "complete": True,
                 "resolved_source_revision": source_revision, "resolved_model_revision": revision,
                 "reference_sha256": file_sha256(stage / "reference.json"),
                 "model_inventory": inventory, "files": _payload_files(stage), "remote_files": remote_files,
                 "verification_scope": "Pinned artifact bytes and layout only; no model loading, inference, or evaluation."}
        _write_json(stage / "acquisition.json", audit)
        verify_acquisition(stage, config_path)
        _freeze(stage)
        if output.exists() or output.is_symlink():
            raise FileExistsError("acquisition output appeared during download")
        stage.rename(output)
        return audit
    except BaseException:
        _remove_stage(stage)
        raise


def verify_acquisition(bundle, config_path=None):
    """Rehash all local files, remote blob identities, pins, and model layout.

    This offline check uses the acquisition's recorded remote identities; it
    does not authenticate a wholly replaced audit. Supply the independently
    maintained reference configuration and retain the acquisition audit hash.
    """
    bundle = Path(bundle)
    files = _payload_files(bundle)
    audit = read_json(bundle / "acquisition.json")
    if (not isinstance(audit, dict) or audit.get("schema_version") != 1
            or audit.get("kind") != "pinned_acquisition" or audit.get("complete") is not True):
        raise ValueError("not a complete acquisition bundle")
    reference = validate_reference(read_json(bundle / "reference.json"))
    if config_path is not None and reference != validate_reference(read_json(config_path)):
        raise ValueError("bundle reference differs from requested configuration")
    if (audit.get("resolved_source_revision") != reference["source_revision"]
            or audit.get("resolved_model_revision") != reference["base_revision"]):
        raise ValueError("recorded resolved revisions disagree with reference")
    if audit.get("reference_sha256") != file_sha256(bundle / "reference.json") or audit.get("files") != files:
        raise ValueError("acquisition payload differs from its audit")
    remote_files = audit.get("remote_files")
    if not isinstance(remote_files, dict) or set(remote_files) != set(files) - {"reference.json"}:
        raise ValueError("remote identities do not cover exactly the acquisition files")
    patch_paths = {"patch/manifest.json": reference["manifest_path"], "patch/edit.npz": reference["archive_path"]}
    if not patch_paths.keys() <= remote_files.keys():
        raise ValueError("selected patch artifacts are missing")
    for name, identity in remote_files.items():
        safe_relative(name)
        if name in patch_paths:
            expected_source, expected_path = "published_patch", patch_paths[name]
        elif name.startswith("model/"):
            expected_source, expected_path = "base_model", name.removeprefix("model/")
        else:
            raise ValueError("unexpected acquisition payload file")
        if not isinstance(identity, dict) or identity.get("source") != expected_source or identity.get("source_path") != expected_path:
            raise ValueError("recorded source path disagrees with the selected artifact")
        _check_identity(bundle / name, identity)
    inventory = inventory_model(bundle / "model")
    if audit.get("model_inventory") != inventory:
        raise ValueError("model inventory differs from saved artifact bytes")
    return audit
