# CI worker and fixtures

The self-hosted Linux worker at `192.168.1.25` uses these labels:

- `self-hosted`, `linux`, `x64`, `unity-linux-cpu`

It runs as a non-privileged account with Docker access. GameCI provides Unity
6000.5.3f1. Configure one supported Unity activation method through repository
secrets:

- Personal: `UNITY_LICENSE` and, when required, `UNITY_EMAIL` and
  `UNITY_PASSWORD`.
- Professional: `UNITY_SERIAL`, `UNITY_EMAIL`, and `UNITY_PASSWORD`.

The workflow stages `ExampleProject~` as `ExampleProject`, then
`run-unity-tests.sh` downloads checksum-pinned ONNX Runtime files and the model
fixtures listed in `Dependencies/fixtures/ci.lock.json`.

Runtime fixtures are stored outside Unity's `Assets` folder under
`ExampleProject/KitsuMateOnnxFixtures`. Only Unity AI Inference models are placed
under `ExampleProject/Assets/KitsuMateOnnxFixtures` so Unity imports them.
Fixtures may not be installed into `Packages`.

There is one required CPU fixture set. There are no smoke, live, release, or
manual test flags. Missing or checksum-mismatched files fail the run. Small
project-authored regression vectors, such as the Kimodo constraint compiler
cases, stay inside their test package; model weights do not.

Production model weights and native runtime binaries are never committed to the
repository. Release jobs download native runtime files before building the
independent package archives.
