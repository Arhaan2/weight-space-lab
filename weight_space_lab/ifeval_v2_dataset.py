"""Offline final-set validation and immutable sealing; no inference capability.

Private curator drafts contain the authored tasks and fixtures. This module
contains only validation, inventory, and freeze logic, never the final text.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import types

from .benchmark_data import load_source, validate_item
from .grading import IFEVAL_PACKAGES, grader_identity, ifeval_metrics, verify_ifeval_assets
from .ifeval_data import ELIGIBILITY_POLICY, SUPPORTED_IDS, TEMPLATE_SEGMENTS, family_coverage, feasibility_reasons, input_tokens, normalize_prompt, sha, stratified_select, write_new
from .protocol import canonical_hash

PINNED_TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
PINNED_WEBLLM_SHA256 = "ae2642e73a4a9896c21efb3a067c4666446f8b49cbe2012f4b576f5d83d5f206"

HISTORICAL_LABEL = "Custom IFEval-style on-device evaluation set using official checkers; not an official Google IFEval split"
LABEL = "Deterministic unexposed subset of pinned official Google IFEval; not the full benchmark"

OFFICIAL_SELECTION = {
    "maximum_prompts": 128,
    "seed": 42,
    "order": "existing rare-instruction coverage selector; stable SHA256 seed/item tie ordering",
    "exposure": "require documented unexposed status; absent, conflicting or unresolved status is excluded",
    "duplicate_rule": "exclude every member of normalized exact or near-duplicate prompt pairs and matches to known exposed prompts",
    "near_duplicate": {"token_set_jaccard_minimum": 0.85, "normalized_character_ratio_minimum": 0.9},
    "shared_instruction_type_is_duplicate": False,
    "selection_observes_model_outputs": False,
}


def load(path):
    return json.loads(Path(path).read_text())


def private_write(path, value):
    write_new(path, value)
    Path(path).chmod(0o600)


def validate_allocation(allocation, items):
    frozen = dict(allocation)
    expected = frozen.pop("sha256")
    if canonical_hash(frozen) != expected:
        raise ValueError("allocation hash mismatch")
    if len(items) != 128 or Counter(len(i["target"]["instruction_id_list"]) for i in items) != {1: 96, 2: 32}:
        raise ValueError("final allocation requires 96 singles and 32 pairs")
    counts = Counter(t for i in items for t in i["target"]["instruction_id_list"])
    if counts != allocation["instruction_type_counts"] or set(counts) != SUPPORTED_IDS or min(counts.values()) < 4:
        raise ValueError("final instruction type allocation mismatch")
    families = Counter()
    for instruction, count in counts.items():
        families[instruction.split(":")[0]] += count
    if families != allocation["instruction_family_counts"] or len(families) != 8 or sum(counts.values()) != 160:
        raise ValueError("final family allocation mismatch")
    singles = Counter(i["target"]["instruction_id_list"][0] for i in items if len(i["target"]["instruction_id_list"]) == 1)
    pairs = Counter(tuple(i["target"]["instruction_id_list"]) for i in items if len(i["target"]["instruction_id_list"]) == 2)
    if singles != allocation["single_type_counts"] or pairs != {tuple(p["instruction_ids"]):p["prompts"] for p in allocation["pair_counts"]}:
        raise ValueError("pre-authored single/pair allocation changed")
    ids, texts = set(), set()
    for item in items:
        validate_item(item)
        if item["split"] != "test" or item["benchmark"] != "ifeval":
            raise ValueError("final role mismatch")
        normalized = normalize_prompt(item["task_text"])
        if item["item_id"] in ids or normalized in texts:
            raise ValueError("duplicate final identity or prompt")
        ids.add(item["item_id"])
        texts.add(normalized)
    return {"prompts": len(items), "instructions":sum(counts.values()), "single_instruction_prompts":96,
            "two_instruction_prompts":32, "instruction_types":dict(sorted(counts.items())),
            "instruction_families":dict(sorted(families.items()))}


def _official_evaluator(root):
    """Use the same pinned official implementation and initialization as grading_worker."""
    verify_ifeval_assets(root)
    for name, version in IFEVAL_PACKAGES.items():
        if importlib.metadata.version(name) != version:
            raise ValueError("pinned evaluator dependency mismatch")
    sys.dont_write_bytecode = True
    root = Path(root).resolve()
    package = types.ModuleType("instruction_following_eval")
    package.__path__ = [str(root / "instruction_following_eval")]
    sys.modules["instruction_following_eval"] = package
    import nltk
    nltk.data.path[:] = [str(root / "nltk_data")]
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
    from instruction_following_eval import evaluation_lib

    def evaluate(item, response):
        random.seed(0)
        example = evaluation_lib.InputExample(key=item["item_id"], prompt=item["task_text"], **item["target"])
        responses = {example.prompt:response}
        strict = evaluation_lib.test_instruction_following_strict(example, responses)
        loose = evaluation_lib.test_instruction_following_loose(example, responses)
        return ifeval_metrics(strict.follow_instruction_list, loose.follow_instruction_list)
    return evaluate


def validate_fixtures(items, fixtures, tokenizer, evaluate):
    if len(fixtures) != len(items) or [f["item_id"] for f in fixtures] != [i["item_id"] for i in items]:
        raise ValueError("fixture identity/order mismatch")
    cases = []
    for item, fixture in zip(items, fixtures):
        size = len(item["target"]["instruction_id_list"])
        count = input_tokens(item["task_text"], tokenizer)
        if feasibility_reasons(item, count):
            raise ValueError("task feasibility failed: " + item["item_id"])
        response = fixture["constructive_response"]
        response_count = len(tokenizer.encode(response, add_special_tokens=False).ids)
        if not 0 < response_count <= 1020:
            raise ValueError("constructive response exceeds sampled output allowance")
        positive = evaluate(item, response)
        if positive["instruction_strict"] != [True] * size or positive["instruction_loose"] != [True] * size:
            raise ValueError("constructive fixture failed: " + item["item_id"])
        negatives = fixture["negative_fixtures"]
        if len(negatives) != size or sorted(n["failed_instruction_index"] for n in negatives) != list(range(size)):
            raise ValueError("one negative fixture per instruction required")
        negative_results = []
        for negative in negatives:
            index = negative["failed_instruction_index"]
            expected = [n != index for n in range(size)]
            result = evaluate(item, negative["response"])
            if result["instruction_strict"] != expected:
                raise ValueError("negative fixture does not isolate intended failure: " + item["item_id"] + ':' + str(index))
            negative_results.append({"failed_instruction_index":index,"expected_strict":expected,"observed":result})
        cases.append({"item_id":item["item_id"],"item_sha256":canonical_hash(item),
                      "target":item["target"],"rendered_input_tokens":count,"constructive_response_tokens":response_count,
                      "constructive_response_sha256":canonical_hash(response),"constructive_result":positive,
                      "combined_constraints_satisfied":True,"negative_results":negative_results})
    return {"schema_version":1,"kind":"ifeval_v2_offline_fixture_validation","all_passed":True,
            "constructive_cases":len(cases),"negative_cases":sum(len(c["negative_results"]) for c in cases),
            "combined_constraint_cases":sum(len(i["target"]["instruction_id_list"]) == 2 for i in items),
            "max_rendered_input_tokens":max(c["rendered_input_tokens"] for c in cases),
            "max_constructive_response_tokens":max(c["constructive_response_tokens"] for c in cases),
            "output_allowance_including_prefix":1024,"maximum_sampled_output_tokens":1020,
            "context_tokens":4096,"target_forward_passes":0,"cases":cases}


def deduplicate(items, inventory):
    """Exact/NFKC-casefold prompt and available underlying-task comparison.

    Inventory includes source-wide data, not only examples known to have runs.
    Lexical similarity is an audit aid, never a non-exposure guarantee.
    """
    historical = inventory["texts"]
    exact = {r["text"] for r in historical}
    normalized = {normalize_prompt(r["text"]) for r in historical}
    sets = [(r["source"],r["id"],set(normalize_prompt(r["text"]).split())) for r in historical]
    cases = []
    for item in items:
        prompt = item["task_text"]
        task = item["metadata"]["underlying_task"]
        if prompt in exact or normalize_prompt(prompt) in normalized or normalize_prompt(task) in normalized:
            raise ValueError("final prompt or underlying task overlaps exposure inventory")
        words = set(normalize_prompt(task).split())
        closest = sorted(((len(words & other)/max(1,len(words|other)),source,key) for source,key,other in sets),reverse=True)[:3]
        cases.append({"item_id":item["item_id"],"exact_matches":0,"normalized_matches":0,"underlying_task_matches":0,
                      "nearest_lexical_matches":[{"jaccard":score,"source":source,"id":key} for score,source,key in closest]})
    return {"schema_version":1,"kind":"ifeval_v2_exposure_deduplication","inventory_sha256":canonical_hash(inventory),
            "inventory_rows":len(historical),"files":inventory["files"],"exact_or_normalized_matches":0,
            "method":"exact plus NFKC casefold whitespace prompt and available underlying-task comparisons; token-set Jaccard nearest three",
            "limitations":["No claim of pretraining non-exposure","Paraphrases and unrecorded exposure remain unverified", "Custody is procedural separation, not verified technical isolation"],"cases":cases}


def seal(root, *, tokenizer_path, ifeval_root, environment_path, bundled_audit_path):
    """Validate a private authored draft and atomically write an immutable manifest last."""
    from tokenizers import Tokenizer
    root = Path(root)
    if (root / "manifest.json").exists():
        raise FileExistsError("final set already frozen; do not overwrite")
    allocation = load(root / "allocation.json")
    draft = load(root / "authored-draft.json")
    provenance = load(root / "provenance.json")
    if draft["authorship_started_epoch_seconds"] <= allocation["frozen_epoch_seconds"]:
        raise ValueError("allocation was not frozen before authorship")
    if provenance["target_forward_passes"] != 0 or provenance["curator_will_propose_weight_edits"] is not False:
        raise ValueError("curator / target-generation separation violated")
    environment = load(environment_path)
    if list(sys.version_info[:3]) != environment["python_version"] or any(importlib.metadata.version(p) != v for p,v in environment["packages"].items()):
        raise ValueError("pinned preparation environment mismatch")
    if importlib.metadata.version("tokenizers") != "0.22.2":
        raise ValueError("pinned tokenizer version mismatch")
    items, fixtures = draft["items"], draft["fixtures"]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for item in items:
        item["metadata"]["rendered_input_tokens"] = input_tokens(item["task_text"], tokenizer)
    counts = validate_allocation(allocation, items)
    validation = validate_fixtures(items, fixtures, tokenizer, _official_evaluator(ifeval_root))
    bundled = load(bundled_audit_path)
    if sha(tokenizer_path) != PINNED_TOKENIZER_SHA256 or bundled["runtime_sha256"] != PINNED_WEBLLM_SHA256:
        raise ValueError("tokenizer or WebLLM runtime differs from frozen V1 configuration")
    if Path(bundled_audit_path).resolve() != (root / "tokenizer-audit.json").resolve():
        raise ValueError("bundled audit must be the frozen evaluator component")
    expected = {c["item_id"]:c["rendered_input_tokens"] for c in validation["cases"]}
    observed = {c["item_id"]:c["rendered_input_tokens"] for c in bundled["counts"]}
    if expected != observed or len(bundled["counts"]) != 128 or bundled["tokenizer_sha256"] != sha(tokenizer_path):
        raise ValueError("bundled tokenizer exact parity failed")
    if bundled["engine_loads"] != 0 or bundled["target_forward_passes"] != 0:
        raise ValueError("token audit used target inference")
    audit = deduplicate(items, load(root / "exposure-inventory.json"))
    # The private curator records interpretation of lexical nearest neighbors.
    review = load(root / "semantic-review.json")
    if review["reviewed_item_ids"] != [i["item_id"] for i in items] or review["unresolved_matches"]:
        raise ValueError("curator semantic/template review incomplete")
    for name, value in [("final-items.json",items),("fixtures.json",fixtures),("validation.json",validation),("dedup-audit.json",audit)]:
        private_write(root / name, value)
    components = ["allocation.json","authored-draft.json","provenance.json","final-items.json","fixtures.json",
                  "validation.json","dedup-audit.json","exposure-inventory.json","semantic-review.json","tokenizer-audit.json",
                  "author-private.py","inventory-private.py","authored-draft-v0.json","draft-revisions.json"]
    manifest = {"schema_version":1,"kind":"ifeval_browser_v2_final_set_freeze","label":HISTORICAL_LABEL,
                "created_epoch_seconds":time.time(),"counts":counts,"ordered_item_ids":[i["item_id"] for i in items],
                "items_sha256":canonical_hash(items),"allocation_sha256":allocation["sha256"],
                "files":{name:sha(root/name) for name in components},"grader":grader_identity("ifeval"),
                "checker_configuration_sha256":canonical_hash([i["target"] for i in items]),
                "tokenizer_sha256":sha(tokenizer_path),"template_sha256":canonical_hash(TEMPLATE_SEGMENTS),
                "environment_sha256":sha(environment_path),"runtime_sha256":bundled["runtime_sha256"],
                "tokenizer_parity":True,"target_forward_passes":0,"editor_isolation":"unverified",
                "custody":"procedural separation; private ignored evaluator store; not verified technical isolation",
                "curator_ai_assistance":True,"final_prompts_in_editor_packets":False,
                "source_sha256":sha(__file__)}
    manifest["sha256"] = canonical_hash(manifest)
    private_write(root / "manifest.json", manifest)
    summary = public_summary(manifest, validation, audit, provenance)
    summary["manifest_file_sha256"] = sha(root / "manifest.json")
    return summary


def public_summary(manifest, validation, audit, provenance):
    """Deliberate allowlist; no source text, answers, private paths, or result rows."""
    return {"schema_version":1,"kind":"ifeval_v2_final_set_public_summary","label":HISTORICAL_LABEL,
            "manifest_sha256":manifest["sha256"],"counts":manifest["counts"],
            "items_sha256":manifest["items_sha256"],"checker_configuration_sha256":manifest["checker_configuration_sha256"],
            "allocation_sha256":manifest["allocation_sha256"],"tokenizer_sha256":manifest["tokenizer_sha256"],
            "template_sha256":manifest["template_sha256"],"runtime_sha256":manifest["runtime_sha256"],
            "fixture_summary":{k:validation[k] for k in ("all_passed","constructive_cases","negative_cases","combined_constraint_cases","max_rendered_input_tokens","max_constructive_response_tokens")},
            "bundled_tokenizer_parity":True,"exposure_inventory_rows":audit["inventory_rows"],
            "exact_or_normalized_matches":audit["exact_or_normalized_matches"],
            "authorship":"AI-assisted authorship in a separate curator agent; no Qwen generations; no claim of independent human authorship",
            "template_reuse":provenance["template_reuse_summary"],"custody":manifest["custody"],
            "editor_isolation":"unverified","target_forward_passes":0,"upstream_attribution":"Google Research IFEval, Zhou et al., Apache-2.0; Qwen tokenizer and WebLLM/MLC retained"}


def verify_frozen(root):
    root = Path(root)
    manifest = load(root / "manifest.json")
    body = dict(manifest)
    digest = body.pop("sha256")
    if canonical_hash(body) != digest or any(sha(root/name) != value for name,value in manifest["files"].items()):
        raise ValueError("frozen final-set component changed")
    if canonical_hash(load(root/"final-items.json")) != manifest["items_sha256"]:
        raise ValueError("frozen ordered items changed")
    return manifest


def _near_duplicate(left, right):
    left, right = normalize_prompt(left), normalize_prompt(right)
    if left == right:
        return True
    a, b = set(re.findall(r"\w+", left)), set(re.findall(r"\w+", right))
    thresholds = OFFICIAL_SELECTION["near_duplicate"]
    return (len(a & b) / max(1, len(a | b)) >= thresholds["token_set_jaccard_minimum"]
            and SequenceMatcher(None, left, right, autojunk=False).ratio()
            >= thresholds["normalized_character_ratio_minimum"])


def select_official(items, exposure, tokenizer, *, maximum=128):
    """Screen task metadata and exposure evidence only; never accept generations.

    The caller supplies an auditable per-item exposure assessment. Stored source
    files and scheduled-but-unexecuted items do not prove exposure or non-exposure.
    A positive unexposed classification needs cited evidence and a completed
    assessment; an absent status remains unresolved rather than becoming clean.
    """
    if type(maximum) is not int or not 0 <= maximum <= 128:
        raise ValueError("official final maximum must be within 0..128")
    items = [deepcopy(validate_item(i)) for i in items]
    if len({i["item_id"] for i in items}) != len(items):
        raise ValueError("duplicate official item identity")
    if any(i["benchmark"] != "ifeval" for i in items):
        raise ValueError("official final set requires IFEval items")
    decisions = exposure.get("items", {})
    duplicates = {i["item_id"]: [] for i in items}
    for index, left in enumerate(items):
        for right in items[index + 1:]:
            if _near_duplicate(left["task_text"], right["task_text"]):
                duplicates[left["item_id"]].append(right["item_id"])
                duplicates[right["item_id"]].append(left["item_id"])
    known_exposed = exposure.get("known_exposed_prompts", [])
    eligible, cases = [], []
    for item in items:
        item["split"] = "test"
        count = input_tokens(item["task_text"], tokenizer)
        item["metadata"]["rendered_input_tokens"] = count
        feasibility = feasibility_reasons(item, count)
        assessment = decisions.get(item["item_id"], {})
        status = assessment.get("status", "unresolved")
        if status not in {"unexposed", "exposed", "unresolved"}:
            raise ValueError("invalid exposure classification")
        if status == "unexposed" and (not assessment.get("evidence")
                                      or assessment.get("assessment_complete") is not True):
            raise ValueError("unexposed classification lacks completed evidence")
        reasons = list(feasibility)
        if status != "unexposed":
            reasons.append("known_exposure" if status == "exposed" else "exposure_unresolved")
        duplicate_matches = duplicates[item["item_id"]]
        exposed_matches = [row["id"] for row in known_exposed
                           if _near_duplicate(item["task_text"], row["text"])]
        if duplicate_matches:
            reasons.append("source_exact_or_near_duplicate")
        if exposed_matches:
            reasons.append("exposed_exact_or_near_duplicate")
        if not reasons:
            eligible.append(item)
        cases.append({"item_id": item["item_id"], "source_item_sha256": canonical_hash(item),
                      "rendered_input_tokens": count, "runtime_eligible": not feasibility,
                      "exposure_status": status, "exposure_evidence": assessment.get("evidence", []),
                      "source_duplicate_ids": sorted(duplicate_matches),
                      "exposed_duplicate_ids": sorted(set(exposed_matches)),
                      "exclusion_reasons": sorted(set(reasons))})
    selected = stratified_select(eligible, min(maximum, len(eligible))) if eligible and maximum else []
    return selected, {
        "schema_version": 1, "kind": "official_ifeval_exposure_and_eligibility_audit",
        "selection": deepcopy(OFFICIAL_SELECTION), "eligibility_policy": deepcopy(ELIGIBILITY_POLICY),
        "source_prompts": len(items), "runtime_eligible_prompts": sum(c["runtime_eligible"] for c in cases),
        "eligible_unexposed_prompts": len(eligible), "selected_prompts": len(selected),
        "exposure_counts": dict(sorted(Counter(c["exposure_status"] for c in cases).items())),
        "exclusion_reason_counts": dict(sorted(Counter(r for c in cases for r in c["exclusion_reasons"]).items())),
        "cases": cases, "target_forward_passes": 0,
        "limitations": ["Known records cannot disprove unlogged historical editor access or pretraining exposure",
                       "Lexical duplicate screening does not prove semantic uniqueness",
                       "Evaluator/editor separation remains procedural; technical isolation is unverified"],
    }


def seal_official(root, *, source_reference, source_cache, exposure_path, tokenizer_path,
                  ifeval_root, environment_path, grading_policy_path):
    """Freeze an audited official subset, including an honest empty final set."""
    from tokenizers import Tokenizer
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("official evaluator directory must be new and empty")
    if sha(tokenizer_path) != PINNED_TOKENIZER_SHA256:
        raise ValueError("tokenizer differs from pinned pristine runtime")
    if importlib.metadata.version("tokenizers") != "0.22.2":
        raise ValueError("pinned tokenizer package mismatch")
    environment = load(environment_path)
    if list(sys.version_info[:3]) != environment["python_version"]:
        raise ValueError("pinned preparation Python mismatch")
    verify_ifeval_assets(ifeval_root)
    for package, version in IFEVAL_PACKAGES.items():
        if importlib.metadata.version(package) != version:
            raise ValueError("pinned evaluator package mismatch")
    reference = load(source_reference)
    source = next(s for s in reference["sources"] if s["key"] == "ifeval")
    items = load_source(source, source_cache)
    exposure = load(exposure_path)
    if exposure.get("source_sha256") != source["sha256"]:
        raise ValueError("exposure inventory refers to a different official source")
    for path, identity in exposure.get("evidence_files", {}).items():
        if identity.get("readable") is False:
            continue
        if sha(path) != identity["sha256"]:
            raise ValueError("exposure evidence changed: " + path)
    selected, audit = select_official(items, exposure, Tokenizer.from_file(str(tokenizer_path)))
    counts = family_coverage(selected)
    counts.update(single_instruction_prompts=sum(len(i["target"]["instruction_id_list"]) == 1 for i in selected),
                  two_instruction_prompts=sum(len(i["target"]["instruction_id_list"]) == 2 for i in selected))
    for name, value in [("final-items.json", selected), ("exposure-inventory.json", exposure),
                        ("selection-audit.json", audit), ("source-reference.json", source),
                        ("grading-policy.json", load(grading_policy_path))]:
        private_write(root / name, value)
    manifest = {
        "schema_version": 2, "kind": "official_ifeval_browser_v2_final_set_freeze", "label": LABEL,
        "created_epoch_seconds": time.time(), "counts": counts, "ordered_item_ids": [i["item_id"] for i in selected],
        "items_sha256": canonical_hash(selected), "source": source,
        "selection": deepcopy(OFFICIAL_SELECTION), "eligibility_policy_sha256": canonical_hash(ELIGIBILITY_POLICY),
        "files": {name: sha(root / name) for name in ("final-items.json", "exposure-inventory.json", "selection-audit.json", "source-reference.json", "grading-policy.json")},
        "grader": grader_identity("ifeval"), "grading_policy_sha256": sha(grading_policy_path),
        "checker_configuration_sha256": canonical_hash([i["target"] for i in selected]),
        "tokenizer_sha256": sha(tokenizer_path), "template_sha256": canonical_hash(TEMPLATE_SEGMENTS),
        "runtime_sha256": PINNED_WEBLLM_SHA256, "environment_sha256": sha(environment_path),
        "source_sha256": sha(__file__), "target_forward_passes": 0,
        "final_quality_available": bool(selected), "editor_isolation": "unverified",
        "custody": "private ignored evaluator store; procedural separation; technical isolation unverified",
        "final_prompts_in_editor_packets": False,
    }
    manifest["sha256"] = canonical_hash(manifest)
    private_write(root / "manifest.json", manifest)
    return official_public_summary(manifest, audit)


def official_public_summary(manifest, audit):
    """Publish counts and identities without evaluator text or exposure records."""
    return {
        "schema_version": 2, "kind": "official_ifeval_v2_final_set_public_summary", "label": LABEL,
        "manifest_sha256": manifest["sha256"], "counts": manifest["counts"],
        "items_sha256": manifest["items_sha256"], "source": manifest["source"],
        "checker_configuration_sha256": manifest["checker_configuration_sha256"],
        "tokenizer_sha256": manifest["tokenizer_sha256"], "template_sha256": manifest["template_sha256"],
        "runtime_sha256": manifest["runtime_sha256"], "selection": manifest["selection"],
        "final_quality_available": manifest["final_quality_available"],
        "audit_counts": {key: audit[key] for key in ("source_prompts", "runtime_eligible_prompts", "eligible_unexposed_prompts", "selected_prompts", "exposure_counts", "exclusion_reason_counts")},
        "limitations": audit["limitations"], "editor_isolation": "unverified", "target_forward_passes": 0,
        "upstream_attribution": "Google Research IFEval, Zhou et al., Apache-2.0; Qwen tokenizer and WebLLM/MLC retained",
    }
