import hashlib
import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location("runtime_validation", Path(__file__).with_name("validate-onnxruntime.py"))
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class RuntimeValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.lock = {"packages": [{"files": [{
            "destination": "Windows/x86_64/onnxruntime.dll",
            "sha256": hashlib.sha256(b"runtime").hexdigest(),
        }]}]}

    def test_directory_checks_content_and_ignores_metadata(self):
        native = self.root / "Windows/x86_64/onnxruntime.dll"
        native.parent.mkdir(parents=True)
        native.write_bytes(b"runtime")
        native.with_suffix(".dll.meta").write_text("Unity metadata")
        self.assertEqual(1, validation.validate(self.lock, root=self.root))
        native.write_bytes(b"wrong version")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            validation.validate(self.lock, root=self.root)

    def test_directory_rejects_missing_and_unexpected_payload(self):
        with self.assertRaisesRegex(ValueError, "missing=.*onnxruntime.dll"):
            validation.validate(self.lock, root=self.root)
        (self.root / "extra.dll").write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "unexpected=.*extra.dll"):
            validation.validate(self.lock, root=self.root)

    def test_archive_checks_actual_bytes(self):
        for payload in (b"runtime", b"wrong version"):
            with self.subTest(payload=payload):
                archive = self.root / "runtime.tgz"
                with tarfile.open(archive, "w:gz") as tar:
                    info = tarfile.TarInfo("package/Runtime/Plugins/Windows/x86_64/onnxruntime.dll")
                    info.size = len(payload)
                    tar.addfile(info, io.BytesIO(payload))
                if payload == b"runtime":
                    self.assertEqual(1, validation.validate(self.lock, archive=archive))
                else:
                    with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                        validation.validate(self.lock, archive=archive)

    def test_archive_rejects_duplicate_entries(self):
        archive = self.root / "runtime.tgz"
        with tarfile.open(archive, "w:gz") as tar:
            for _ in range(2):
                info = tarfile.TarInfo("package/Runtime/Plugins/Windows/x86_64/onnxruntime.dll")
                info.size = 7
                tar.addfile(info, io.BytesIO(b"runtime"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validation.validate(self.lock, archive=archive)


if __name__ == "__main__":
    unittest.main()
