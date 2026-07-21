# CI bootstrap contract

The self-hosted Linux worker at `192.168.1.25` must be registered with labels:

- `self-hosted`, `linux`, `x64`, `unity-linux-cpu`

Run it as a dedicated non-privileged runner account with Docker access. GameCI pulls
the compatible Unity 6000.5.3f1 image on the first test run, so no manually
preinstalled Unity editor image is required. Repository settings must restrict this
runner group to protected `main` pushes, version tags, and manual dispatches; it
must not accept fork or pull-request jobs.

`unity-integration.yml` runs GameCI's pinned `game-ci/unity-test-runner@v4`
explicitly for the `ExampleProject~` EditMode and PlayMode suites. Configure one
supported Unity activation method as repository secrets before dispatching it:

- Personal: `UNITY_LICENSE` (and, when required by the license, `UNITY_EMAIL` and
  `UNITY_PASSWORD`).
- Professional: `UNITY_SERIAL`, `UNITY_EMAIL`, and `UNITY_PASSWORD`.

The workflow grants `checks: write` so GameCI can publish test checks and uploads
test-result artifacts even when Unity fails. Do not add back a generic
`KITSUMATE_UNITY_TEST_COMMAND` variable: arbitrary command evaluation is deliberately
not part of this workflow.

Before Unity starts, `run-unity-tests.sh` has these responsibilities:

1. Download the checksum-pinned ONNX Runtime 1.24.4 artifact and the required external CPU CI fixture set into the runner cache.
2. Verify checksums and install fixtures only beneath `ExampleProject~/Assets/KitsuMateOnnxFixtures`; fixture installation into `Packages/` is rejected.
3. Stage the inputs in the checked-out Unity example project for the GameCI
   container to consume.
4. Let GameCI run the requested core/backend/feature test assemblies and publish
   its NUnit XML artifacts.

The required fixture set is the committed `Dependencies/fixtures/ci.lock.json` file with this form:

```json
{
  "files": [
    {
      "id": "onnxruntime-cpu",
      "url": "https://example.invalid/onnxruntime-cpu.onnx",
      "sha256": "<lowercase SHA-256>",
      "destination": "ExampleProject~/Assets/KitsuMateOnnxFixtures/onnxruntime/smoke.onnx"
    }
  ]
}
```

The single CI fixture set must also provide:

- `ExampleProject~/Assets/KitsuMateOnnxFixtures/unity-ai-inference/smoke.onnx`
- `ExampleProject~/Assets/KitsuMateOnnxFixtures/metadata/yolo10n_external.onnx`
- `ExampleProject~/Assets/KitsuMateOnnxFixtures/motion/KimodoConstraints/` and its manifest/case files

There is no release, smoke, live, or manually selected variant. Engine, metadata, and conditioning integration tests fail—not skip—when their required fixture is absent, so fixture setup cannot be mistaken for a passing integration result.

Models and native runtime binaries must never be committed to a UPM package or the repository. The release workflow will package each production UPM directory into its own `.tgz` after this validation step.
