"""Pinned benchmark acquisition, identity-preserving splits, and data bundles.

Bundle labels describe intended audiences; they are not access controls.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import tempfile
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from .protocol import canonical_hash


_BENCHMARKS = {"math", "gsm8k", "ifeval"}
_SPLITS = {"development", "validation", "test"}
_ITEM_FIELDS = {"benchmark", "item_id", "dataset_revision", "split", "task_text",
                "target", "source", "metadata"}
_SPLIT_POLICY = {
    "id": "template-groups-v1", "selection_validation_modulus": 5,
    "ifeval_partition": "family_hash_mod_5:0=test,1=validation,2-4=development",
    "final_math": "all MATH-500 rows", "final_gsm8k": "all original GSM8K test rows",
    "ifeval_scope": "study-specific held-out subset; not the complete official 541-prompt benchmark",
}


def _read(path):
    from .__main__ import read_json
    return read_json(Path(path))


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_item(item):
    """Validate a canonical source item without inferring missing identities."""
    if not isinstance(item, dict) or set(item) != _ITEM_FIELDS:
        raise ValueError("benchmark item has missing or unknown fields")
    if item["benchmark"] not in _BENCHMARKS or item["split"] not in _SPLITS:
        raise ValueError("invalid benchmark or split")
    for key in ("item_id", "task_text"):
        if not isinstance(item[key], str) or not item[key].strip():
            raise ValueError(f"item {key} must be a nonempty string")
    if not isinstance(item["dataset_revision"], str) or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", item["dataset_revision"]):
        raise ValueError("dataset revision must be an immutable commit")
    source = item["source"]
    if not isinstance(source, dict) or set(source) != {"dataset_id", "revision", "split", "row_id"}:
        raise ValueError("invalid source identity")
    if any(not isinstance(v, str) or not v for v in source.values()) or source["revision"] != item["dataset_revision"]:
        raise ValueError("source identity disagrees with dataset revision")
    if not isinstance(item["metadata"], dict) or not isinstance(item["target"], dict):
        raise ValueError("metadata and target must be objects")
    target = item["target"]
    if item["benchmark"] == "ifeval":
        if set(target) != {"instruction_id_list", "kwargs"}:
            raise ValueError("invalid instruction target")
        ids, kwargs = target["instruction_id_list"], target["kwargs"]
        if not isinstance(ids, list) or not ids or any(not isinstance(v, str) or not v for v in ids):
            raise ValueError("instruction identities must be nonempty strings")
        if not isinstance(kwargs, list) or len(ids) != len(kwargs) or any(not isinstance(v, dict) for v in kwargs):
            raise ValueError("instruction arguments must align with instruction identities")
    elif set(target) != {"answer"} or not isinstance(target["answer"], str) or not target["answer"].strip():
        raise ValueError("mathematical target requires an explicit answer")
    canonical_hash(item)  # Also rejects non-JSON or nonfinite metadata.
    return copy.deepcopy(item)


def _reference(reference):
    ref = _read(reference) if isinstance(reference, (str, Path)) else copy.deepcopy(reference)
    if not isinstance(ref, dict) or set(ref) != {"schema_version", "kind", "sources", "split_policy", "licenses"}:
        raise ValueError("invalid benchmark source reference")
    if type(ref["schema_version"]) is not int or ref["schema_version"] != 1 or ref["kind"] != "benchmark_sources":
        raise ValueError("invalid benchmark source reference version")
    keys = set()
    if not isinstance(ref["sources"], list) or not ref["sources"]:
        raise ValueError("source reference must contain files")
    for spec in ref["sources"]:
        expected = {"key", "benchmark", "dataset_id", "revision", "path", "format", "source_split",
                    "row_count", "role", "bytes", "sha256", "url"}
        if not isinstance(spec, dict) or set(spec) != expected:
            raise ValueError("invalid source file specification")
        if not isinstance(spec["key"], str) or not re.fullmatch(r"[a-z0-9_-]+", spec["key"]) or spec["key"] in keys:
            raise ValueError("source keys must be unique safe filenames")
        keys.add(spec["key"])
        if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]) or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]):
            raise ValueError("source must pin immutable revision and exact bytes")
        if any(type(spec[k]) is not int or spec[k] <= 0 for k in ("bytes", "row_count")):
            raise ValueError("source size and row count must be positive integers")
        if spec["benchmark"] not in _BENCHMARKS or spec["format"] not in {"parquet", "jsonl"} or spec["role"] not in {"selection", "final", "partition"}:
            raise ValueError("unsupported benchmark source")
        expected_urls = [f"https://huggingface.co/datasets/{spec['dataset_id']}/resolve/{spec['revision']}/{spec['path']}",
                         f"https://raw.githubusercontent.com/{spec['dataset_id']}/{spec['revision']}/{spec['path']}"]
        if spec["url"] not in expected_urls or ".." in Path(spec["path"]).parts or Path(spec["path"]).is_absolute():
            raise ValueError("source URL must identify its pinned source file")
    if ref["split_policy"] != _SPLIT_POLICY:
        raise ValueError("unsupported split policy")
    return ref


def _verify_file(path, spec):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != spec["bytes"]:
        raise ValueError("source file size or type does not match pin")
    data = path.read_bytes()
    if _sha(data) != spec["sha256"]:
        raise ValueError("source file hash does not match pin")
    return data


def acquire_benchmarks(reference, output, *, max_download_bytes=12_000_000, timeout=30):
    """Bounded public downloads; publish a cache only after every pin verifies."""
    ref, output = _reference(reference), Path(output)
    if type(max_download_bytes) is not int or max_download_bytes <= 0 or type(timeout) not in (int, float) or not 0 < timeout <= 300:
        raise ValueError("download bounds must be positive and timeout at most 300 seconds")
    total = sum(s["bytes"] for s in ref["sources"])
    if total > max_download_bytes:
        raise ValueError("pinned sources exceed download budget")
    if output.exists():
        raise ValueError("benchmark cache exists; choose a new output")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".benchmark-download-", dir=output.parent))
    try:
        for spec in ref["sources"]:
            deadline = time.monotonic() + timeout
            with urllib.request.urlopen(spec["url"], timeout=timeout) as response, (temporary / spec["key"]).open("xb") as handle:
                received = 0
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError("benchmark download exceeded elapsed-time budget")
                    reader = getattr(response, "read1", response.read)
                    chunk = reader(min(65536, spec["bytes"] - received + 1))
                    if time.monotonic() > deadline:
                        raise TimeoutError("benchmark download exceeded elapsed-time budget")
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > spec["bytes"]:
                        raise ValueError("download exceeds pinned file size")
                    handle.write(chunk)
            _verify_file(temporary / spec["key"], spec)
        summary = {"kind": "benchmark_acquisition", "source_manifest_sha256": canonical_hash(ref),
                   "verified_files": len(ref["sources"]), "verified_bytes": total,
                   "experiment_executed": False}
        _write(temporary / "acquisition.json", summary)
        temporary.rename(output)
        return summary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _source_answer(solution):
    """Extract the last balanced source solution box, without grading output."""
    starts = list(re.finditer(r"\\(?:boxed|fbox)(?![a-zA-Z])\s*", solution))
    if not starts:
        raise ValueError("source MATH solution has no boxed answer")
    start = starts[-1].end()
    if start == len(solution):
        raise ValueError("source MATH solution has malformed final box")
    if solution[start] != "{":
        single = re.match(r"([0-9])(?=\$|\s|[.,]|$)", solution[start:])
        if single:
            return single[1]
        raise ValueError("source MATH solution has malformed final box")
    start += 1
    depth = 1
    for index in range(start, len(solution)):
        char = solution[index]
        backslashes = 0
        prior = index - 1
        while prior >= 0 and solution[prior] == "\\":
            backslashes += 1
            prior -= 1
        if backslashes % 2:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                answer = solution[start:index]
                if answer.strip():
                    return answer
                break
    raise ValueError("source MATH solution has malformed final box")


def load_source(spec, cache_root, *, source_diagnostics=None):
    """Read verified source bytes and preserve exact upstream row identities."""
    data = _verify_file(Path(cache_root) / spec["key"], spec)
    if spec["format"] == "parquet":
        try:
            import pyarrow
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ValueError("MATH parquet loading requires optional pyarrow==19.0.1") from error
        if pyarrow.__version__ != "19.0.1":
            raise ValueError("MATH parquet loading requires pinned pyarrow==19.0.1")
        rows = pq.read_table(pyarrow.BufferReader(data)).to_pylist()
    else:
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != spec["row_count"]:
        raise ValueError("source row count does not match pin")
    result = []
    for index, row in enumerate(rows):
        benchmark = spec["benchmark"]
        metadata = {"source_file": spec["path"], "source_file_sha256": spec["sha256"], "source_row_index": index}
        if benchmark == "math":
            if "unique_id" in row:
                item_id = row["unique_id"]
                answer = row["answer"]
            else:
                # The parquet mirror does not publish original competition filenames.
                item_id = f"{spec['path']}:{index:06d}"
                try:
                    answer = _source_answer(row["solution"])
                except ValueError:
                    if spec["role"] != "selection" or source_diagnostics is None:
                        raise
                    source_diagnostics.append({"benchmark": benchmark, "item_id": item_id,
                                               "reason": "unresolved_source_answer", "source_key": spec["key"],
                                               "source_file_sha256": spec["sha256"], "source_row_index": index})
                    continue
            prompt, target = row["problem"], {"answer": answer}
            metadata.update(subject=row.get("subject", row.get("type")), level=row.get("level"),
                            source_solution=row["solution"])
        elif benchmark == "gsm8k":
            item_id = f"{spec['source_split']}:{index:06d}"
            prompt = row["question"]
            answer = row["answer"].rsplit("####", 1)
            if len(answer) != 2 or not answer[1].strip():
                raise ValueError("GSM8K source lacks its final-answer delimiter")
            target = {"answer": answer[1].strip()}
            metadata["source_solution"] = row["answer"]
        else:
            if type(row["key"]) is not int:
                raise ValueError("IFEval source key must be an integer")
            item_id, prompt = str(row["key"]), row["prompt"]
            target = {"instruction_id_list": row["instruction_id_list"], "kwargs": row["kwargs"]}
        source = {"dataset_id": spec["dataset_id"], "revision": spec["revision"],
                  "split": spec["source_split"], "row_id": item_id}
        item = {"benchmark": benchmark, "item_id": item_id, "dataset_revision": spec["revision"],
                "split": "test" if spec["role"] == "final" else "development", "task_text": prompt,
                "target": target, "source": source, "metadata": metadata}
        result.append(validate_item(item))
    if len({item["item_id"] for item in result}) != len(result):
        raise ValueError("duplicate source item identity")
    return result


def to_generation_tasks(items):
    """Project exact task text to native-runner tasks, with no hidden wrapping."""
    checked = [validate_item(item) for item in items]
    return [{key: item[key] for key in ("benchmark", "item_id", "split", "dataset_revision")} |
            {"messages": [{"role": "user", "content": item["task_text"]}]} for item in checked]


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def template_family(item):
    """Conservative exact skeletons, not a semantic deduplication claim."""
    if item["benchmark"] == "ifeval":
        # Hold out instruction combinations, including instruction multiplicity.
        skeleton = sorted(item["target"]["instruction_id_list"])
    else:
        skeleton = unicodedata.normalize("NFKC", item["task_text"])
        if item["benchmark"] == "gsm8k":
            # Proper names/capitalized words and numbers may instantiate a template.
            skeleton = re.sub(r"\b[A-Z][a-z]+\b", "NAME", skeleton)
        skeleton = normalized_text(skeleton)
        skeleton = re.sub(r"\d+(?:[.,]\d+)*", "#", skeleton)
        if item["benchmark"] == "math":
            skeleton = re.sub(r"(?<![a-z\\])[a-z](?![a-z])", "VAR", skeleton)
        skeleton = re.sub(r"\s+", " ", skeleton).strip()
    return canonical_hash({"benchmark": item["benchmark"], "skeleton": skeleton})


def audit_overlap(items):
    """Report cross-split exact or skeleton families without emitting task text."""
    checked = [validate_item(item) for item in items]
    identifiers, exact, families = defaultdict(list), defaultdict(list), defaultdict(list)
    for item in checked:
        identity = {"benchmark": item["benchmark"], "item_id": item["item_id"], "split": item["split"]}
        identifiers[(item["benchmark"], item["item_id"])].append(identity)
        exact[(item["benchmark"], _sha(normalized_text(item["task_text"]).encode()))].append(identity)
        families[(item["benchmark"], template_family(item))].append(identity)
    duplicate_ids = [v for v in identifiers.values() if len(v) > 1]
    exact_cross = [v for v in exact.values() if len({r["split"] for r in v}) > 1]
    family_cross = [v for v in families.values() if len({r["split"] for r in v}) > 1]
    return {"schema_version": 1, "method": "template-groups-v1", "item_count": len(items),
            "duplicate_identities": duplicate_ids, "cross_split_exact": exact_cross,
            "cross_split_template_families": family_cross,
            "passed": not (duplicate_ids or exact_cross or family_cross),
            "limitations": "Exact normalized and skeleton matches only; paraphrases and training contamination remain unproven."}


def partition_items(source_items):
    """Freeze whole template families before any model output is inspected.

    source_items consists of (role, item) pairs, where role is selection,
    final, or partition. Selection duplicates of fixed final tasks are
    quarantined rather than removing final tasks from the denominator.
    """
    final = [validate_item(item) for role, item in source_items if role == "final"]
    final_ids = {(i["benchmark"], i["item_id"]) for i in final}
    final_text = {(i["benchmark"], normalized_text(i["task_text"])) for i in final}
    final_families = {(i["benchmark"], template_family(i)) for i in final}
    selected, quarantine = [], []
    for role, original in source_items:
        item = validate_item(original)
        if role not in {"selection", "final", "partition"}:
            raise ValueError("unknown source role")
        family = template_family(item)
        if role == "selection":
            reason = ("source_identity" if (item["benchmark"], item["item_id"]) in final_ids else
                      "normalized_text" if (item["benchmark"], normalized_text(item["task_text"])) in final_text else
                      "template_family" if (item["benchmark"], family) in final_families else None)
            if reason:
                quarantine.append({"benchmark": item["benchmark"], "item_id": item["item_id"], "reason": reason,
                                   "source": copy.deepcopy(item["source"]), "metadata": copy.deepcopy(item["metadata"])})
                continue
            item["split"] = "validation" if int(family, 16) % 5 == 0 else "development"
        elif role == "partition":
            if item["benchmark"] != "ifeval":
                raise ValueError("only IFEval uses study-specific held-out partitions")
            bucket = int(family, 16) % 5
            item["split"] = {0: "test", 1: "validation"}.get(bucket, "development")
        else:
            item["split"] = "test"
        selected.append(item)
    selected.sort(key=lambda item: (item["benchmark"], item["item_id"]))
    audit = audit_overlap(selected)
    audit["quarantined_selection_items"] = quarantine
    if not audit["passed"]:
        raise ValueError("split overlap audit failed; no bundle prepared")
    return selected, audit


_AUDIENCES = {"development": "editor-development", "validation": "evaluator-validation", "test": "evaluator-final"}


def prepare_benchmark_data(reference, cache, output):
    """Create separate audience bundles and explicit identity manifests."""
    ref, output = _reference(reference), Path(output)
    if output.exists():
        raise ValueError("benchmark bundle exists; choose a new output")
    source_diagnostics = []
    source_items = [(spec["role"], item) for spec in ref["sources"]
                    for item in load_source(spec, cache, source_diagnostics=source_diagnostics)]
    items, audit = partition_items(source_items)
    audit["source_quality_quarantine"] = source_diagnostics
    counts = {benchmark: {split: sum(i["benchmark"] == benchmark and i["split"] == split for i in items)
                          for split in sorted(_SPLITS)} for benchmark in sorted(_BENCHMARKS)}
    if any(count == 0 for splits in counts.values() for count in splits.values()):
        raise ValueError("each benchmark requires nonempty development, validation and final-test splits")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".benchmark-prepare-", dir=output.parent))
    try:
        hashes = {}
        for split, audience in _AUDIENCES.items():
            subset = [i for i in items if i["split"] == split]
            manifest = {"schema_version": 1, "kind": "benchmark_split", "split": split,
                        "source_manifest_sha256": canonical_hash(ref),
                        "items": [{"benchmark": i["benchmark"], "item_id": i["item_id"],
                                   "dataset_revision": i["dataset_revision"], "item_sha256": canonical_hash(i)} for i in subset]}
            _write(temporary / audience / "items.json", subset)
            _write(temporary / audience / "tasks.json", to_generation_tasks(subset))
            _write(temporary / audience / "manifest.json", manifest)
            hashes[split] = canonical_hash(manifest)
        _write(temporary / "source-reference.json", ref)
        _write(temporary / "overlap-audit.json", audit)
        summary = {"schema_version": 1, "kind": "benchmark_data_bundle", "source_manifest_sha256": canonical_hash(ref),
                   "counts": counts, "split_manifest_sha256": hashes, "overlap_audit_sha256": canonical_hash(audit),
                   "quarantined_selection_items": len(audit["quarantined_selection_items"]),
                   "source_quality_quarantine": len(source_diagnostics),
                   "ifeval_scope": ref["split_policy"]["ifeval_scope"], "access_boundary_enforced": False,
                   "isolation_prerequisite": "Phase 7 must enforce evaluator/editor filesystem and tool access separation.",
                   "experiment_executed": False}
        _write(temporary / "bundle.json", summary)
        check_benchmark_bundle(temporary)
        temporary.rename(output)
        return summary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def check_benchmark_bundle(output, *, cache=None):
    """Check internal identity/coverage; optional cache replays source conversion."""
    output = Path(output)
    ref = _reference(_read(output / "source-reference.json"))
    summary = _read(output / "bundle.json")
    if summary["kind"] != "benchmark_data_bundle" or summary["source_manifest_sha256"] != canonical_hash(ref):
        raise ValueError("bundle source identity mismatch")
    all_items = []
    for split, audience in _AUDIENCES.items():
        items = [validate_item(i) for i in _read(output / audience / "items.json")]
        manifest = _read(output / audience / "manifest.json")
        if canonical_hash(manifest) != summary["split_manifest_sha256"][split]:
            raise ValueError("split manifest digest mismatch")
        expected = {"schema_version": 1, "kind": "benchmark_split", "split": split,
                    "source_manifest_sha256": canonical_hash(ref),
                    "items": [{"benchmark": i["benchmark"], "item_id": i["item_id"],
                               "dataset_revision": i["dataset_revision"], "item_sha256": canonical_hash(i)} for i in items]}
        if expected != manifest or any(i["split"] != split for i in items):
            raise ValueError("bundle items differ from frozen split manifest")
        if _read(output / audience / "tasks.json") != to_generation_tasks(items):
            raise ValueError("generation tasks differ from canonical task text")
        all_items.extend(items)
    counts = {benchmark: {split: sum(i["benchmark"] == benchmark and i["split"] == split for i in all_items)
                          for split in sorted(_SPLITS)} for benchmark in sorted(_BENCHMARKS)}
    audit = _read(output / "overlap-audit.json")
    if canonical_hash(audit) != summary["overlap_audit_sha256"] or counts != summary["counts"]:
        raise ValueError("bundle counts or audit digest mismatch")
    if summary["quarantined_selection_items"] != len(audit["quarantined_selection_items"]) or summary["source_quality_quarantine"] != len(audit["source_quality_quarantine"]):
        raise ValueError("bundle quarantine count mismatch")
    if any(count <= 0 for splits in counts.values() for count in splits.values()):
        raise ValueError("bundle splits must remain nonempty")
    specifications = {(s["dataset_id"], s["revision"], s["path"]): s for s in ref["sources"]}
    coverage = defaultdict(set)

    def register(spec, index, digest):
        if spec is None or digest != spec["sha256"] or type(index) is not int or not 0 <= index < spec["row_count"]:
            raise ValueError("item source file identity is outside pinned catalogue")
        if index in coverage[spec["key"]]:
            raise ValueError("source row is represented more than once")
        coverage[spec["key"]].add(index)

    for item in all_items:
        spec = specifications.get((item["source"]["dataset_id"], item["dataset_revision"], item["metadata"].get("source_file")))
        register(spec, item["metadata"].get("source_row_index"), item["metadata"].get("source_file_sha256"))
        if item["source"]["split"] != spec["source_split"] or item["source"]["row_id"] != item["item_id"]:
            raise ValueError("item source row identity mismatch")
        bucket = int(template_family(item), 16) % 5
        expected_split = ("test" if spec["role"] == "final" else
                          ("validation" if bucket == 0 else "development") if spec["role"] == "selection" else
                          {0: "test", 1: "validation"}.get(bucket, "development"))
        if item["split"] != expected_split:
            raise ValueError("item split differs from predetermined source-role policy")
    for row in audit["quarantined_selection_items"]:
        source, metadata = row["source"], row["metadata"]
        spec = specifications.get((source["dataset_id"], source["revision"], metadata.get("source_file")))
        register(spec, metadata.get("source_row_index"), metadata.get("source_file_sha256"))
        if spec["role"] != "selection" or row["reason"] not in {"source_identity", "normalized_text", "template_family"}:
            raise ValueError("only selection overlap may be quarantined")
    by_key = {s["key"]: s for s in ref["sources"]}
    for row in audit["source_quality_quarantine"]:
        spec = by_key.get(row["source_key"])
        register(spec, row["source_row_index"], row["source_file_sha256"])
        if spec["role"] != "selection" or row["reason"] != "unresolved_source_answer":
            raise ValueError("invalid source quality quarantine")
    if any(len(coverage[s["key"]]) != s["row_count"] for s in ref["sources"]):
        raise ValueError("bundle must account for every source row")
    recalculated = audit_overlap(all_items)
    if any(recalculated[k] != audit[k] for k in recalculated) or not recalculated["passed"]:
        raise ValueError("bundle overlap audit mismatch")
    if summary["access_boundary_enforced"] is not False or summary["experiment_executed"] is not False:
        raise ValueError("data preparation cannot claim isolation or experiment execution")
    if cache is not None:
        diagnostics = []
        source_items = [(s["role"], i) for s in ref["sources"]
                        for i in load_source(s, cache, source_diagnostics=diagnostics)]
        expected_items, expected_audit = partition_items(source_items)
        expected_audit["source_quality_quarantine"] = diagnostics
        if sorted(all_items, key=lambda i: (i["benchmark"], i["item_id"])) != expected_items or audit != expected_audit:
            raise ValueError("bundle differs from verified source bytes")
    return summary
