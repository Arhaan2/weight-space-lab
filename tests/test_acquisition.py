import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from weight_space_lab.acquisition import (
    _HTTP, acquire_published, validate_reference, verify_acquisition,
)


def encoded(value):
    return json.dumps(value).encode()


def git_sha(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


class Response(io.BytesIO):
    def __init__(self, data, url):
        super().__init__(data)
        self.url = url

    def geturl(self):
        return self.url

    def read1(self, length):
        return self.read(length)


class PinnedServer:
    """Synthetic HTTPS fixtures; no network or real model evidence."""

    def __init__(self):
        self.reference = {
            "schema_version": 1, "claim_type": "published_patch_replay",
            "source_repository": "https://github.com/example/release",
            "source_revision": "1" * 40, "base_model": "example/model",
            "base_revision": "2" * 40, "manifest_path": "release/manifest.json",
            "archive_path": "release/edit.npz", "tensor_names": ["weight"],
            "expected_changed_stored_entries": 1, "reported_targeted_entries": 1,
        }
        header = encoded({"weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}})
        header += b" " * (-len(header) % 8)
        self.shard = "model-00001-of-00001.safetensors"
        self.model_files = {
            "config.json": encoded({"model_type": "synthetic"}),
            "tokenizer_config.json": encoded({"chat_template": "{{ messages }}"}),
            "tokenizer.json": encoded({"synthetic": True}),
            "model.safetensors.index.json": encoded({"weight_map": {"weight": self.shard}, "metadata": {"total_size": 4}}),
            self.shard: len(header).to_bytes(8, "little") + header + bytes.fromhex("803f0000"),
            "LICENSE": b"Synthetic upstream license notice.\n",
            "video_preprocessor_config.json": encoded({"synthetic": True}),
            "conversation.json": b"never download",
            "README.md": b"never download",
            "alternative.safetensors": b"never download",
            "pytorch_model.bin": b"never download",
        }
        self.patch_files = {"release/manifest.json": b'{"synthetic": true}', "release/edit.npz": b"synthetic patch"}
        self.urls = {}
        self.calls = []
        self.source_prefix = "https://api.github.com/repos/example/release"
        self.source_pin = self.reference["source_revision"]
        self.model_pin = self.reference["base_revision"]
        self.commit_url = f"{self.source_prefix}/git/commits/{self.source_pin}"
        self.urls[self.commit_url] = encoded({"sha": self.source_pin})
        for name, data in self.patch_files.items():
            meta_url = f"{self.source_prefix}/contents/{name}?ref={self.source_pin}"
            self.urls[meta_url] = encoded({"type": "file", "path": name, "size": len(data), "sha": git_sha(data)})
            self.urls[f"https://raw.githubusercontent.com/example/release/{self.source_pin}/{name}"] = data
        siblings = []
        for name, data in self.model_files.items():
            info = {"rfilename": name, "size": len(data), "blobId": git_sha(data)}
            if name == self.shard:
                info["lfs"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            siblings.append(info)
            self.urls[self.model_url(name)] = data
        self.model_meta_url = f"https://huggingface.co/api/models/example/model/revision/{self.model_pin}?blobs=true"
        self.urls[self.model_meta_url] = encoded({"sha": self.model_pin, "siblings": siblings})

    def model_url(self, name):
        return f"https://huggingface.co/example/model/resolve/{self.model_pin}/{name}"

    def open(self, request, timeout):
        self.calls.append(request.full_url)
        if request.full_url not in self.urls:
            raise AssertionError(f"unexpected HTTP request: {request.full_url}")
        return Response(self.urls[request.full_url], request.full_url)


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.server = PinnedServer()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "reference.json"
        self.config.write_text(json.dumps(self.server.reference))
        self.output = self.root / "bundle"
        self.opener = patch("weight_space_lab.acquisition.urllib.request.build_opener", return_value=self.server)
        self.opener.start()
        self.addCleanup(self.opener.stop)

    def acquire(self, **kwargs):
        return acquire_published(self.config, self.output, **kwargs)

    def test_selected_snapshot_complete_and_read_only(self):
        audit = self.acquire()
        self.assertEqual(verify_acquisition(self.output, self.config), audit)
        self.assertEqual(audit["model_inventory"]["parameter_count"], 2)
        self.assertEqual(audit["resolved_model_revision"], self.server.model_pin)
        self.assertEqual(audit["remote_files"][f"model/{self.server.shard}"]["algorithm"], "sha256")
        self.assertEqual(audit["remote_files"]["model/config.json"]["algorithm"], "git_sha1")
        for excluded in ("conversation.json", "README.md", "alternative.safetensors", "pytorch_model.bin"):
            self.assertNotIn(self.server.model_url(excluded), self.server.calls)
            self.assertFalse((self.output / "model" / excluded).exists())
        self.assertEqual((self.output / "model/LICENSE").read_bytes(), self.server.model_files["LICENSE"])
        self.assertEqual(self.output.stat().st_mode & 0o222, 0)
        self.assertTrue(all(path.stat().st_mode & 0o222 == 0 for path in self.output.rglob("*")))
        self.assertNotIn(str(self.root), json.dumps(audit))

    def test_moving_pins_bad_paths_and_counts_rejected_before_network(self):
        for field, value in (("base_revision", "main"), ("source_revision", "v1"),
                             ("manifest_path", "../secret"), ("archive_path", "/absolute"),
                             ("reported_targeted_entries", True), ("tensor_names", ["x", "x"]),
                             ("source_repository", "http://github.com/example/release")):
            with self.subTest(field=field):
                bad = {**self.server.reference, field: value}
                with self.assertRaises(ValueError):
                    validate_reference(bad)
        self.assertEqual(self.server.calls, [])

    def test_resolved_revision_mismatch_never_publishes(self):
        for url in (self.server.commit_url, self.server.model_meta_url):
            original = self.server.urls[url]
            data = json.loads(original)
            data["sha"] = "3" * 40
            self.server.urls[url] = encoded(data)
            with self.assertRaises(ValueError):
                self.acquire()
            self.assertFalse(self.output.exists())
            self.assertEqual(list(self.root.glob(".acquisition-incomplete-*")), [])
            self.server.urls[url] = original

    def test_corrupted_downloads_and_wrong_size_never_publish(self):
        for name, suffix in ((self.server.shard, b"x"), ("config.json", b"")):
            url = self.server.model_url(name)
            original = self.server.urls[url]
            self.server.urls[url] = (bytes([original[0] ^ 1]) + original[1:]) + suffix
            with self.assertRaises(ValueError):
                self.acquire()
            self.assertFalse(self.output.exists())
            self.server.urls[url] = original

    def test_budget_limits_and_failed_download_cleanup(self):
        for limit in (1, 1000):
            with self.assertRaises(ValueError):
                self.acquire(max_download_bytes=limit)
            self.assertFalse(self.output.exists())
        with patch.object(self.server, "open", side_effect=TimeoutError("fixture timeout")):
            with self.assertRaises(TimeoutError):
                self.acquire()
        self.assertEqual(list(self.root.glob(".acquisition-incomplete-*")), [])

    def test_missing_shard_and_unsafe_index_rejected(self):
        url = self.server.model_url("model.safetensors.index.json")
        for shard in ("missing.safetensors", "../escape.safetensors"):
            index = encoded({"weight_map": {"weight": shard}})
            self.server.urls[url] = index
            meta = json.loads(self.server.urls[self.server.model_meta_url])
            for item in meta["siblings"]:
                if item["rfilename"] == "model.safetensors.index.json":
                    item.update(size=len(index), blobId=git_sha(index))
            self.server.urls[self.server.model_meta_url] = encoded(meta)
            with self.assertRaises(ValueError):
                self.acquire()
            self.assertFalse(self.output.exists())

    def test_reverification_detects_tampering_added_files_and_wrong_reference(self):
        self.acquire()
        wrong = {**self.server.reference, "base_revision": "3" * 40}
        self.config.write_text(json.dumps(wrong))
        with self.assertRaises(ValueError):
            verify_acquisition(self.output, self.config)
        asset = self.output / "model/config.json"
        asset.chmod(0o644)
        original = asset.read_bytes()
        asset.write_bytes(original + b" ")
        with self.assertRaises(ValueError):
            verify_acquisition(self.output)
        asset.write_bytes(original)
        self.output.chmod(0o755)
        extra = self.output / "unexpected.json"
        extra.write_text("{}")
        with self.assertRaises(ValueError):
            verify_acquisition(self.output)
        extra.unlink()
        extra.symlink_to(self.config)
        with self.assertRaises(ValueError):
            verify_acquisition(self.output)

    def test_existing_output_refused_without_new_network_calls(self):
        self.acquire()
        calls = len(self.server.calls)
        with self.assertRaises(FileExistsError):
            self.acquire()
        self.assertEqual(len(self.server.calls), calls)

    def test_remote_identity_cannot_cover_extra_payload_or_wrong_source(self):
        self.acquire()
        path = self.output / "acquisition.json"
        path.chmod(0o644)
        audit = json.loads(path.read_text())
        audit["remote_files"]["patch/manifest.json"]["source_path"] = "unselected.json"
        path.write_text(json.dumps(audit))
        with self.assertRaises(ValueError):
            verify_acquisition(self.output)

    def test_http_budget_deadline_and_transport_validation(self):
        for values in ((True, 1, 1), (10, float("inf"), 1), (10, 1, 0)):
            with self.assertRaises(ValueError):
                _HTTP(*values)
        http = _HTTP(100, 1, 1)
        with self.assertRaises(ValueError):
            http.download("http://example.invalid/data", self.root / "unsafe")
        with patch("weight_space_lab.acquisition.time.monotonic", return_value=http.deadline + 1):
            with self.assertRaises(TimeoutError):
                http.download(self.server.commit_url, self.root / "expired")

    def test_http_existing_file_is_preserved(self):
        path = self.root / "keep.json"
        path.write_bytes(b"preserve existing file")
        with self.assertRaises(FileExistsError):
            _HTTP(1000, 1).download(self.server.commit_url, path)
        self.assertEqual(path.read_bytes(), b"preserve existing file")
        self.assertEqual(self.server.calls, [])

    def test_http_checks_deadline_after_read_including_eof(self):
        clock = [0.0]

        class DelayedResponse(Response):
            def read(self, length):
                clock[0] = 2.0
                return super().read(length)

        for data in (b"", b"late data"):
            with self.subTest(data=data):
                clock[0] = 0.0
                destination = self.root / "late"
                response = DelayedResponse(data, self.server.commit_url)
                with patch("weight_space_lab.acquisition.time.monotonic", side_effect=lambda: clock[0]), \
                        patch.object(self.server, "open", return_value=response):
                    with self.assertRaises(TimeoutError):
                        _HTTP(100, 1).download(self.server.commit_url, destination)
                self.assertFalse(destination.exists())

    def test_http_allows_eof_exactly_at_deadline(self):
        clock = [0.0]

        class BoundaryResponse(Response):
            def read(self, length):
                clock[0] = 1.0
                return super().read(length)

        response = BoundaryResponse(b"", self.server.commit_url)
        destination = self.root / "boundary"
        with patch("weight_space_lab.acquisition.time.monotonic", side_effect=lambda: clock[0]), \
                patch.object(self.server, "open", return_value=response):
            _HTTP(100, 1).download(self.server.commit_url, destination, size=0)
        self.assertEqual(destination.read_bytes(), b"")

    def test_http_uses_incremental_read1(self):
        class IncrementalResponse(Response):
            calls = 0

            def read(self, length):
                raise AssertionError("buffer-filling read must not be used")

            def read1(self, length):
                self.calls += 1
                return io.BytesIO.read(self, min(length, 2))

        response = IncrementalResponse(b"fixture", self.server.commit_url)
        destination = self.root / "incremental"
        with patch.object(self.server, "open", return_value=response):
            _HTTP(100, 1).download(self.server.commit_url, destination, size=7)
        self.assertEqual(destination.read_bytes(), b"fixture")
        self.assertEqual(response.calls, 5)


if __name__ == "__main__":
    unittest.main()
