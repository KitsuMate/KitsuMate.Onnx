# CI fixtures

The Unity job requires Docker and Unity 6000.5.3f1. Configure one supported
Unity activation method through repository secrets:

- Personal: `UNITY_LICENSE` and, when required, `UNITY_EMAIL` and
  `UNITY_PASSWORD`.
- Professional: `UNITY_SERIAL`, `UNITY_EMAIL`, and `UNITY_PASSWORD`.

The workflow stages `ExampleProject~` as `ExampleProject`, then
`run-unity-tests.sh` downloads checksum-pinned ONNX Runtime files and the model
fixtures listed in `Dependencies/fixtures/ci.lock.json`.

The job keeps content-addressed artifacts and models below
`RUNNER_TOOL_CACHE`. A fixture is fetched only when its SHA-256-named cache file
is absent. Clean checkouts copy verified files from this persistent cache
instead of downloading them again. The workflow does not use `actions/cache`
for model weights because the fixture set is large and already persists in the
tool cache.

Runtime fixtures are stored outside Unity's `Assets` folder under
`ExampleProject/KitsuMateOnnxFixtures`. Only Unity AI Inference models are placed
under `ExampleProject/Assets/KitsuMateOnnxFixtures` so Unity imports them.
Fixtures may not be installed into `Packages`.

There is one required CPU fixture set. There are no smoke, live, release, or
manual test flags. Missing or checksum-mismatched files fail the run. Small
project-authored regression vectors, such as the Kimodo constraint compiler
cases, stay inside their test package; model weights do not.

Production model weights and native runtime binaries are never committed to the
repository. Release jobs provision and validate the default backend and NVIDIA
provider package independently from `onnxruntime.lock.json` and
`onnxruntime-nvidia.lock.json`; a file may belong to only one lock/package.

The Windows DirectML core is source-built from the pinned ONNX Runtime commit.
Before tagging a release, publish the checksum-matching archive named in the
main lock. CUDA provider plug-ins are built with the scripts under
`Tools/native`; TensorRT-RTX consumes its separately pinned standalone EP ABI
artifact. Archive validation rejects missing, extra, mismatched, and
cross-package native files.
