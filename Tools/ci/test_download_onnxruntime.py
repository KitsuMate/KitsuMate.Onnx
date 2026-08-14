import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("download-onnxruntime.py")


class ArtifactHydrationTests(unittest.TestCase):
    def test_nested_extraction_preserves_meta_and_removes_stale_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "runtime.aar"
            with zipfile.ZipFile(nested, "w") as archive:
                archive.writestr("jni/arm64-v8a/libonnxruntime.so", b"native")
            package = root / "package.nupkg"
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("runtimes/android/native/onnxruntime.aar", nested.read_bytes())
                archive.writestr("lib/netstandard2.0/runtime.dll", b"managed")

            destination = root / "output"
            destination.mkdir()
            (destination / "tracked.meta").write_text("guid: retained", encoding="utf-8")
            (destination / "stale.bin").write_bytes(b"stale")
            files = [
                {
                    "archive": "runtimes/android/native/onnxruntime.aar",
                    "source": "jni/arm64-v8a/libonnxruntime.so",
                    "destination": "Android/arm64-v8a/libonnxruntime.so",
                    "sha256": hashlib.sha256(b"native").hexdigest(),
                },
                {
                    "source": "lib/netstandard2.0/runtime.dll",
                    "destination": "Managed/runtime.dll",
                    "sha256": hashlib.sha256(b"managed").hexdigest(),
                },
            ]
            lock = root / "lock.json"
            lock.write_text(json.dumps({
                "version": "test",
                "packages": [{"id": "fixture", "url": package.as_uri(), "files": files}],
            }), encoding="utf-8")

            completed = subprocess.run([
                sys.executable, str(SCRIPT), "--lock", str(lock),
                "--cache", str(root / "cache"), "--destination", str(destination),
            ], capture_output=True, text=True)
            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertEqual(b"native", (destination / files[0]["destination"]).read_bytes())
            self.assertEqual(b"managed", (destination / files[1]["destination"]).read_bytes())
            self.assertTrue((destination / "tracked.meta").is_file())
            self.assertFalse((destination / "stale.bin").exists())

    def test_rejects_destination_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "package.nupkg"
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("value", b"data")
            lock = root / "lock.json"
            lock.write_text(json.dumps({
                "version": "test",
                "packages": [{"id": "fixture", "url": package.as_uri(), "files": [{
                    "source": "value", "destination": "../escaped",
                    "sha256": hashlib.sha256(b"data").hexdigest(),
                }]}],
            }), encoding="utf-8")
            completed = subprocess.run([
                sys.executable, str(SCRIPT), "--lock", str(lock),
                "--cache", str(root / "cache"), "--destination", str(root / "output"),
            ], capture_output=True, text=True)
            self.assertNotEqual(0, completed.returncode)
            self.assertFalse((root / "escaped").exists())

    def test_recovers_corrupt_cached_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "source.nupkg"
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("value", b"data")
            cache = root / "cache"
            cache.mkdir()
            (cache / "fixture.test.nupkg").write_bytes(b"not a zip")
            lock = root / "lock.json"
            lock.write_text(json.dumps({
                "version": "test",
                "packages": [{"id": "fixture", "url": package.as_uri(), "files": [{
                    "source": "value", "destination": "value.bin",
                    "sha256": hashlib.sha256(b"data").hexdigest(),
                }]}],
            }), encoding="utf-8")
            completed = subprocess.run([
                sys.executable, str(SCRIPT), "--lock", str(lock),
                "--cache", str(cache), "--destination", str(root / "output"),
            ], capture_output=True, text=True)
            self.assertEqual(0, completed.returncode, completed.stderr)


if __name__ == "__main__":
    unittest.main()
