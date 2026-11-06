import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from weight_space_lab.benchmark_data import (
    _reference, _source_answer, acquire_benchmarks, audit_overlap,
    check_benchmark_bundle, load_source, normalized_text, partition_items,
    prepare_benchmark_data, template_family, to_generation_tasks, validate_item,
)


REVISION = "a" * 40


def item(benchmark="math", item_id="fixture-a", text="Find the roots of a polynomial.", split="development"):
    target = {"instruction_id_list": ["fixture:one"], "kwargs": [{}]} if benchmark == "ifeval" else {"answer": "2"}
    return {"benchmark": benchmark, "item_id": item_id, "dataset_revision": REVISION, "split": split,
            "task_text": text, "target": target,
            "source": {"dataset_id": "fixture/source", "revision": REVISION, "split": "train", "row_id": item_id},
            "metadata": {}}


def fixture_reference(root):
    """Independently authored source data, never benchmark score evidence."""
    reference = json.loads((Path(__file__).parents[1] / "configs/benchmark_sources.json").read_text())
    reference["sources"] = []
    names = ["amber", "birch", "cedar", "dahlia", "elm", "fir", "garden", "hazel", "iris", "juniper", "kelp", "larch"]
    for benchmark in ("math", "gsm8k", "ifeval"):
        roles = ["partition"] if benchmark == "ifeval" else ["selection", "final"]
        for role in roles:
            rows = []
            for n, name in enumerate(names if role != "final" else ["zinnia"]):
                if benchmark == "math":
                    rows.append({"unique_id": f"{role}/{name}", "problem": f"Find the {name} polynomial roots.",
                                 "answer": "2", "solution": r"An independently prepared fixture: \boxed{2}.", "subject": "fixture", "level": 1})
                elif benchmark == "gsm8k":
                    rows.append({"question": f"How many {name} leaves are there?", "answer": "Two fixture leaves.\n#### 2"})
                else:
                    rows.append({"key": n, "prompt": f"Write a {name} sentence.", "instruction_id_list": [f"fixture:{name}"], "kwargs": [{}]})
            key = benchmark + "-" + role
            data = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
            (root / key).write_bytes(data)
            path = key + ".jsonl"
            reference["sources"].append({"key": key, "benchmark": benchmark, "dataset_id": "fixture/source",
                                         "revision": REVISION, "path": path, "format": "jsonl", "source_split": role,
                                         "row_count": len(rows), "role": role, "bytes": len(data),
                                         "sha256": hashlib.sha256(data).hexdigest(),
                                         "url": f"https://raw.githubusercontent.com/fixture/source/{REVISION}/{path}"})
    return reference


class BenchmarkIdentityTests(unittest.TestCase):
    def test_item_validation_requires_source_and_preserves_original_id(self):
        original = item(item_id="test/algebra/42.json")
        self.assertEqual(validate_item(original), original)
        modified = validate_item(original)
        modified["metadata"]["new"] = True
        self.assertEqual(original["metadata"], {})
        for field, value in (("dataset_revision", "main"), ("item_id", ""), ("task_text", ""), ("target", {"answer": ""})):
            changed = copy.deepcopy(original)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_item(changed)

    def test_source_box_extraction_nested_equivalent_and_malformed(self):
        self.assertEqual(_source_answer(r"Work \boxed{1} final \boxed{\frac{2}{4}}."), r"\frac{2}{4}")
        self.assertEqual(_source_answer(r"Answer $\boxed 2$."), "2")
        self.assertEqual(_source_answer(r"Earlier \boxed{9}, finally $\boxed 2$."), "2")
        self.assertEqual(_source_answer(r"Final \boxed{\{1,2\}}."), r"\{1,2\}")
        for text in ("no answer", r"\boxed{}", r"\boxed{2", r"\boxed{9} then \boxed", r"\boxed{9} then \boxed{}"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                _source_answer(text)

    def test_task_projection_preserves_prompt_and_has_no_targets(self):
        original = item(text="Exact  whitespace\nwithout wrapping.")
        tasks = to_generation_tasks([original])
        self.assertEqual(tasks[0]["messages"], [{"role": "user", "content": original["task_text"]}])
        self.assertEqual(set(tasks[0]), {"benchmark", "item_id", "split", "dataset_revision", "messages"})

    def test_math_numeric_and_variable_template_family(self):
        first = item(text="Solve $2x + 3 = 7$.")
        second = item(text="Solve $4y + 5 = 9$.")
        self.assertEqual(template_family(first), template_family(second))

    def test_gsm_template_masks_names_and_numbers(self):
        first = item("gsm8k", text="Alice has 12 apples. How many remain?")
        second = item("gsm8k", text="Robert has 54 apples. How many remain?")
        self.assertEqual(template_family(first), template_family(second))

    def test_ifeval_template_uses_instruction_multiset_not_prompt(self):
        first = item("ifeval", text="Write about a mountain.")
        second = item("ifeval", text="Write about the sea.")
        self.assertEqual(template_family(first), template_family(second))
        second["target"] = {"instruction_id_list": ["fixture:one", "fixture:one"], "kwargs": [{}, {}]}
        self.assertNotEqual(template_family(first), template_family(second))

    def test_overlap_reports_independent_ids_exact_and_template(self):
        first = item("gsm8k", "a", "Alice has 12 apples.")
        second = item("gsm8k", "b", "Robert has 18 apples.", "test")
        report = audit_overlap([first, second])
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["cross_split_template_families"]), 1)
        self.assertFalse(report["cross_split_exact"])
        second["task_text"] = "ALICE  has 12 apples."
        self.assertEqual(len(audit_overlap([first, second])["cross_split_exact"]), 1)
        second["item_id"] = "a"
        self.assertEqual(len(audit_overlap([first, second])["duplicate_identities"]), 1)
        self.assertEqual(normalized_text("Ａ  B\n"), "a b")

    def test_final_overlap_quarantines_selection_never_final(self):
        final = item("math", "test/source.json", "Solve $3x + 7 = 10$.", "test")
        selection = item("math", "train/file.parquet:000003", "Solve $5y + 9 = 14$.")
        rows, audit = partition_items([("selection", selection), ("final", final)])
        self.assertEqual(rows, [final])
        self.assertEqual(audit["quarantined_selection_items"][0]["reason"], "template_family")

    def test_partition_is_order_independent_and_keeps_families_together(self):
        first = item("gsm8k", "train:000000", "Alice has 12 apples.")
        second = item("gsm8k", "train:000001", "Robert has 18 apples.")
        rows, audit = partition_items([("selection", first), ("selection", second)])
        reverse, _ = partition_items([("selection", second), ("selection", first)])
        self.assertEqual(rows, reverse)
        self.assertEqual(rows[0]["split"], rows[1]["split"])
        self.assertTrue(audit["passed"])


class BenchmarkBundleTests(unittest.TestCase):
    def test_pins_source_identities_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = fixture_reference(root)
            spec = next(s for s in reference["sources"] if s["key"] == "gsm8k-selection")
            rows = load_source(spec, root)
            self.assertEqual(rows[0]["item_id"], "selection:000000")
            self.assertEqual(rows[0]["source"]["row_id"], rows[0]["item_id"])
            self.assertEqual(rows[0]["target"]["answer"], "2")
            data = (root / spec["key"]).read_bytes()
            (root / spec["key"]).write_bytes(b" " + data[1:])
            with self.assertRaisesRegex(ValueError, "hash"):
                load_source(spec, root)

    def test_source_reference_rejects_moving_pins_duplicate_keys_and_wrong_urls(self):
        with tempfile.TemporaryDirectory() as temp:
            reference = fixture_reference(Path(temp))
            for field, value in (("revision", "main"), ("sha256", "a" * 40), ("url", "https://example.org/data"), ("bytes", True)):
                changed = copy.deepcopy(reference)
                changed["sources"][0][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    _reference(changed)
            reference["sources"].append(copy.deepcopy(reference["sources"][0]))
            with self.assertRaises(ValueError):
                _reference(reference)

    def test_acquisition_verifies_all_bytes_and_is_append_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = fixture_reference(root)
            payloads = {s["url"]: (root / s["key"]).read_bytes() for s in reference["sources"]}
            with patch("urllib.request.urlopen", side_effect=lambda url, **_: io.BytesIO(payloads[url])):
                report = acquire_benchmarks(reference, root / "acquired")
            self.assertEqual(report["verified_files"], len(payloads))
            self.assertFalse(report["experiment_executed"])
            with self.assertRaisesRegex(ValueError, "exists"):
                acquire_benchmarks(reference, root / "acquired")
            with self.assertRaisesRegex(ValueError, "budget"):
                acquire_benchmarks(reference, root / "too-small", max_download_bytes=1)

    def test_acquisition_cleans_failed_partial_and_rejects_overlong_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = fixture_reference(root)
            spec = reference["sources"][0]
            oversized = (root / spec["key"]).read_bytes() + b"extra"
            with patch("urllib.request.urlopen", return_value=io.BytesIO(oversized)):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    acquire_benchmarks(reference, root / "bad")
            self.assertFalse((root / "bad").exists())
            self.assertFalse(list(root.glob(".benchmark-download-*")))

    def test_bundle_roundtrip_audience_contents_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = fixture_reference(root)
            summary = prepare_benchmark_data(reference, root, root / "bundle")
            self.assertEqual(check_benchmark_bundle(root / "bundle"), summary)
            self.assertFalse(summary["access_boundary_enforced"])
            self.assertFalse(summary["experiment_executed"])
            for audience, split in (("editor-development", "development"), ("evaluator-validation", "validation"), ("evaluator-final", "test")):
                rows = json.loads((root / "bundle" / audience / "items.json").read_text())
                self.assertTrue(rows)
                self.assertEqual({row["split"] for row in rows}, {split})
            path = root / "bundle" / "evaluator-final" / "items.json"
            rows = json.loads(path.read_text())
            rows[0]["target"]["answer"] = "wrong replacement"
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "manifest"):
                check_benchmark_bundle(root / "bundle")

    def test_generation_task_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = fixture_reference(root)
            prepare_benchmark_data(reference, root, root / "bundle")
            path = root / "bundle" / "editor-development" / "tasks.json"
            tasks = json.loads(path.read_text())
            tasks[0]["messages"][0]["content"] += " Hidden prompt wrapping."
            path.write_text(json.dumps(tasks))
            with self.assertRaisesRegex(ValueError, "generation tasks"):
                check_benchmark_bundle(root / "bundle")

    def test_optional_cache_rechecks_source_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = fixture_reference(root)
            summary = prepare_benchmark_data(reference, root, root / "bundle")
            self.assertEqual(check_benchmark_bundle(root / "bundle", cache=root), summary)
            source_path = root / reference["sources"][0]["key"]
            data = source_path.read_bytes()
            source_path.write_bytes(b" " + data[1:])
            with self.assertRaisesRegex(ValueError, "hash"):
                check_benchmark_bundle(root / "bundle", cache=root)

    def test_resealed_missing_source_row_is_rejected(self):
        from weight_space_lab.protocol import canonical_hash
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = fixture_reference(root)
            prepare_benchmark_data(reference, root, root / "bundle")
            bundle = root / "bundle"
            summary = json.loads((bundle / "bundle.json").read_text())
            audience = bundle / "editor-development"
            items = json.loads((audience / "items.json").read_text())
            removed = items.pop()
            (audience / "items.json").write_text(json.dumps(items))
            (audience / "tasks.json").write_text(json.dumps(to_generation_tasks(items)))
            manifest = json.loads((audience / "manifest.json").read_text())
            manifest["items"].pop()
            (audience / "manifest.json").write_text(json.dumps(manifest))
            summary["split_manifest_sha256"]["development"] = canonical_hash(manifest)
            summary["counts"][removed["benchmark"]]["development"] -= 1
            audit = json.loads((bundle / "overlap-audit.json").read_text())
            audit["item_count"] -= 1
            (bundle / "overlap-audit.json").write_text(json.dumps(audit))
            summary["overlap_audit_sha256"] = canonical_hash(audit)
            (bundle / "bundle.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "every source row"):
                check_benchmark_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
