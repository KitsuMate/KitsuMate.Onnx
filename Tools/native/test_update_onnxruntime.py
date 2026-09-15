import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("update-onnxruntime.py")


class ArtifactProvisioningTests(unittest.TestCase):
    def test_locked_runtime_prevents_partial_replacement(self):
        import importlib.util
        from unittest.mock import patch
        spec = importlib.util.spec_from_file_location("runtime_updater", SCRIPT)
        updater = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(updater)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            for name in ("managed.dll", "native.dll"):
                (output / name).write_bytes(b"old")
            lock = root / "lock.json"
            lock.write_text('{"packages": []}')

            def stage(_, args):
                for name in ("managed.dll", "native.dll"):
                    (args.destination / name).write_bytes(b"new")

            original_open = Path.open
            def open_file(path, mode="r", *args, **kwargs):
                if path == output / "native.dll" and mode == "r+b":
                    raise PermissionError("native DLL is loaded")
                return original_open(path, mode, *args, **kwargs)

            with patch.object(updater, "provision_payload", stage), patch.object(Path, "open", open_file), patch.object(sys, "argv", [
                str(SCRIPT), "--lock", str(lock), "--cache", str(root / "cache"), "--destination", str(output)
            ]):
                with self.assertRaises(PermissionError):
                    updater.main()
            self.assertEqual(b"old", (output / "managed.dll").read_bytes())
            self.assertEqual(b"old", (output / "native.dll").read_bytes())

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
            (cache / "fixture.test.zip").write_bytes(b"not a zip")
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

    def test_download_failure_names_package_and_url(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_package = (root / "missing.nupkg").as_uri()
            destination = root / "output"
            destination.mkdir()
            existing = destination / "value.bin"
            existing.write_bytes(b"installed runtime")
            lock = root / "lock.json"
            lock.write_text(json.dumps({
                "version": "test",
                "packages": [{"id": "missing-runtime", "url": missing_package, "files": [{
                    "source": "value", "destination": "value.bin",
                    "sha256": hashlib.sha256(b"data").hexdigest(),
                }]}],
            }), encoding="utf-8")
            completed = subprocess.run([
                sys.executable, str(SCRIPT), "--lock", str(lock),
                "--cache", str(root / "cache"), "--destination", str(root / "output"),
            ], capture_output=True, text=True)
            self.assertNotEqual(0, completed.returncode)
            self.assertIn("missing-runtime", completed.stderr)
            self.assertIn(missing_package, completed.stderr)
            self.assertEqual(b"installed runtime", existing.read_bytes())



if __name__ == "__main__":
    unittest.main()
