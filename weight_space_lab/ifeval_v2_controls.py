"""Frozen V2 support randomization and development selection. Arhaan2.

These pure functions use no weights, model calls, or candidate outcomes when
constructing a control. V1 checkpoint/proposal schemas are left unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

from .protocol import canonical_hash


TENSORS = tuple(f"model.language_model.layers.{layer}.mlp.down_proj.weight" for layer in (15, 19))
ROWS, WIDTH, BLOCK_WIDTH, GROUP_SIZE = 1024, 3584, 60, 32
FACTORS = (0.5, 0.75, 1.25, 1.5)
CONTROL_SEED = 20261108
BASELINE_PASSES = 16
MAPPING = {
    "quantization": "q4f16_1", "linear_weight_layout": "NK", "group_axis": 1,
    "group_size": GROUP_SIZE, "source_shape": [ROWS, WIDTH],
    "q_weight_shape": [ROWS, WIDTH // 8], "q_scale_shape": [ROWS, WIDTH // GROUP_SIZE],
    "source_to_deployed": "direct row/column mapping; language_model prefix removed; float16 cast",
    "group_identity": "(tensor, row, column // 32)",
    "packed_word": "(row, column // 8), nibble shift 4 * (column % 8)",
    "upstream_revision": "9fa644f54b04983adea4d0168f49fc6af4a893ba",
    "mapping_sources": ["python/mlc_llm/model/qwen35/qwen35_loader.py",
        "python/mlc_llm/quantization/quantization.py",
        "python/mlc_llm/quantization/group_quantization.py"],
    "mapping_source_sha256": {
        "qwen35_loader.py": "67c7d0e9b285438a6d5c92e508346d9916c0c782cf758fb212497679bdbb1dd1",
        "quantization.py": "037513b75e009379033dda61b0ba3e44f015c798f767c4bbfe7862a82203b84b",
        "group_quantization.py": "69553f348d222eb3c22fce2c4b57bcda182e172ee2f912bc53c8d9cab7a09542",
    },
}


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_v2_proposal(proposal, base_sha256, *, provenance="editor_proposal"):
    """Validate the frozen narrow search, then retain the original operation order.

This supplements, never replaces, saved-byte checkpoint validation. Direction
removal is supported by the shared schema but is outside this V2 study.
"""
    if (not isinstance(proposal, dict) or set(proposal) != {
            "schema_version", "base_sha256", "provenance", "composition", "operations"}
            or type(proposal["schema_version"]) is not int or proposal["schema_version"] != 1
            or not _digest(base_sha256) or proposal["base_sha256"] != base_sha256
            or provenance not in ("editor_proposal", "engineering_edit")
            or proposal["provenance"] != provenance or proposal["composition"] != "sequential"):
        raise ValueError("invalid V2 proposal identity, schema, provenance, or composition")
    operations = proposal["operations"]
    if not isinstance(operations, list) or len(operations) != 2:
        raise ValueError("V2 requires exactly two scale_columns operations")
    seen = set()
    for operation in operations:
        if (not isinstance(operation, dict) or set(operation) != {"op", "tensor", "columns", "factor"}
                or operation["op"] != "scale_columns"
                or not isinstance(operation["tensor"], str) or operation["tensor"] not in TENSORS
                or operation["tensor"] in seen):
            raise ValueError("V2 requires one scale_columns operation per fixed tensor")
        seen.add(operation["tensor"])
        columns = operation["columns"]
        if (not isinstance(columns, list) or len(columns) != BLOCK_WIDTH
                or any(type(column) is not int for column in columns)
                or not 0 <= columns[0] <= WIDTH - BLOCK_WIDTH
                or columns != list(range(columns[0], columns[0] + BLOCK_WIDTH))):
            raise ValueError("V2 columns must be one ordered contiguous block of 60")
        factor = operation["factor"]
        if type(factor) not in (int, float) or not math.isfinite(factor) or factor not in FACTORS:
            raise ValueError("V2 scale factor is outside the frozen discrete set")
    return deepcopy(proposal)


def support_profile(start):
    """Actual per-row group occupancy, including partially occupied edge groups."""
    if type(start) is not int or not 0 <= start <= WIDTH - BLOCK_WIDTH:
        raise ValueError("invalid block start")
    counts = {}
    for column in range(start, start + BLOCK_WIDTH):
        group = column // GROUP_SIZE
        counts[group] = counts.get(group, 0) + 1
    return {
        "start": start, "start_modulo_group": start % GROUP_SIZE,
        "groups_per_row": len(counts), "occupied_row_groups": ROWS * len(counts),
        "selected_coordinates": ROWS * BLOCK_WIDTH,
        "selected_coordinates_per_group_per_row": list(counts.values()),
        "groups": [{"column_group": group, "selected_per_row": count,
                    "selected_across_rows": count * ROWS} for group, count in counts.items()],
    }


def generate_control(proposal, round_number):
    """Return a reproducible engineering edit and an honest matched-control audit.

The SHA-256 ordering is fixed pseudorandomization, not an entropy-backed draw.
Only the identical paired start is excluded. Partial pair overlap and overlap
with other rounds are allowed; neither scores nor prior supports cause redraws.
"""
    if type(round_number) is not int or round_number not in range(1, 5):
        raise ValueError("round must be one of the four frozen slots")
    proposal = validate_v2_proposal(proposal, proposal.get("base_sha256") if isinstance(proposal, dict) else None)
    control = deepcopy(proposal)
    control["provenance"] = "engineering_edit"
    matching = {}
    for operation in control["operations"]:
        tensor, start = operation["tensor"], operation["columns"][0]
        eligible = list(range(start % GROUP_SIZE, WIDTH - BLOCK_WIDTH + 1, GROUP_SIZE))
        eligible.remove(start)
        ranked = sorted((canonical_hash({"seed": CONTROL_SEED, "round": round_number,
            "tensor": tensor, "start": other}), other) for other in eligible)
        selected_digest, selected = ranked[0]
        operation["columns"] = list(range(selected, selected + BLOCK_WIDTH))
        astra_profile, control_profile = support_profile(start), support_profile(selected)
        for key in ("start_modulo_group", "groups_per_row", "occupied_row_groups",
                    "selected_coordinates", "selected_coordinates_per_group_per_row"):
            if astra_profile[key] != control_profile[key]:
                raise AssertionError("quantization footprint matching failed")
        matching[tensor] = {
            "astra": astra_profile, "control": control_profile,
            "inherited_factor": operation["factor"], "eligible_starts": eligible,
            "selected_rank_digest": selected_digest,
            "overlapping_source_columns": max(0, BLOCK_WIDTH - abs(start - selected)),
            "overlapping_column_groups": sorted({g["column_group"] for g in astra_profile["groups"]}
                & {g["column_group"] for g in control_profile["groups"]}),
        }
    audit = {
        "schema_version": 1, "kind": "ifeval_browser_v2_matched_control",
        "provenance": "matched_random_control", "round": round_number, "seed": CONTROL_SEED,
        "algorithm": "lowest SHA256 of compact sorted-key JSON {seed,round,tensor,start}; integer start breaks ties",
        "mapping": deepcopy(MAPPING), "astra_proposal": proposal,
        "astra_proposal_sha256": canonical_hash(proposal), "control_proposal_sha256": canonical_hash(control),
        "matching": matching, "uses_candidate_scores": False,
        "inherited": ["tensor", "row support", "block width", "per-tensor scale factor", "group alignment"],
        "randomized": ["valid different block start in each tensor"],
        "overlap_rule": "paired identical start excluded; partial and across-round overlaps permitted; no redraw",
        "not_matched": ["source L2 change", "saved changed coordinates", "decoded changed coordinates", "decoded L2 change"],
    }
    return control, audit


def verify_control(proposal, audit):
    if not isinstance(audit, dict) or "astra_proposal" not in audit or "round" not in audit:
        raise ValueError("missing matched-control provenance")
    expected, expected_audit = generate_control(audit["astra_proposal"], audit["round"])
    if proposal != expected or audit != expected_audit:
        raise ValueError("matched control or deterministic provenance changed")
    return deepcopy(audit)


def select_candidate(candidates, baseline_artifact_sha256):
    """Identical rule for each method: strictly beat baseline, earliest round ties."""
    if not _digest(baseline_artifact_sha256) or not isinstance(candidates, list):
        raise ValueError("invalid selection input")
    selected = {"artifact_sha256": baseline_artifact_sha256, "round": None,
                "strict_passes": BASELINE_PASSES, "selection": "pristine_fallback"}
    seen, artifact_scores = set(), {}
    for candidate in candidates:
        if not isinstance(candidate, dict) or type(candidate.get("round")) is not int:
            raise ValueError("invalid selection round record")
    for candidate in sorted(candidates, key=lambda row: row["round"]):
        round_number = candidate["round"]
        if type(round_number) is not int or round_number not in range(1, 5) or round_number in seen:
            raise ValueError("selection requires at most one record per frozen round")
        seen.add(round_number)
        if not all(candidate.get(key) is True for key in ("completed", "technically_valid", "independently_graded")):
            continue
        if candidate.get("scheduled_prompts") != 32 or candidate.get("graded_prompts") != 32:
            continue
        passes = candidate.get("strict_passes")
        if type(passes) is not int or not 0 <= passes <= 32 or not _digest(candidate.get("artifact_sha256")):
            raise ValueError("invalid completed candidate selection record")
        identity = candidate["artifact_sha256"]
        if identity in artifact_scores and artifact_scores[identity] != passes:
            raise ValueError("identical artifact has inconsistent frozen development grades")
        artifact_scores[identity] = passes
        if candidate["artifact_sha256"] == baseline_artifact_sha256:
            if passes != BASELINE_PASSES:
                raise ValueError("identical pristine artifact cannot have different frozen development grades")
            continue
        if passes > selected["strict_passes"]:
            selected = {"artifact_sha256": candidate["artifact_sha256"], "round": round_number,
                        "strict_passes": passes, "selection": "development_strict_improvement"}
    return selected


def final_arms(baseline_artifact_sha256, astra_selection, control_selection):
    """Freeze three named arms onto at most three distinct dispatch identities."""
    named = [("pristine_baseline", baseline_artifact_sha256),
             ("selected_astra", astra_selection["artifact_sha256"]),
             ("selected_random_control", control_selection["artifact_sha256"])]
    distinct = []
    for name, identity in named:
        if not _digest(identity):
            raise ValueError("invalid final artifact identity")
        existing = next((row for row in distinct if row["artifact_sha256"] == identity), None)
        if existing is None:
            distinct.append({"artifact_sha256": identity, "dispatch_arm": name, "named_arms": [name]})
        else:
            existing["named_arms"].append(name)
    return {"named_arms": dict(named), "distinct_evaluations": distinct,
            "unused_final_slots_expire": 3 - len(distinct)}
