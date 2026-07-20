# CI bootstrap contract

The self-hosted Linux worker at `192.168.1.25` must be registered with labels:

- `self-hosted`, `linux`, `x64`, `unity-linux-cpu`

Run it as a dedicated non-privileged runner account. It must expose Docker and a GameCI-compatible Unity 6000.5.3f1 image. Repository settings must restrict this runner group to protected `main` pushes, version tags, and manual dispatches; it must not accept fork or pull-request jobs.

The protected runner must define the repository variable `KITSUMATE_UNITY_TEST_COMMAND`. It must be a reviewed GameCI invocation that runs the `ExampleProject~` EditMode and PlayMode suites and writes NUnit XML under `TestResults/`. The workflow intentionally fails if this variable is not configured; it must never silently claim Unity validation passed.

Before Unity starts, `run-unity-tests.sh` has these responsibilities:

1. Download the checksum-pinned ONNX Runtime 1.24.4 artifact and the selected external fixture profile into the runner cache.
2. Verify checksums and install artifacts only into the workspace staging directories.
3. Mount the staged artifacts into the GameCI container.
4. Run the requested core/backend/feature test assemblies and publish NUnit XML.

Models and native runtime binaries must never be committed to a UPM package or the repository. The release workflow will package each production UPM directory into its own `.tgz` after this validation step.
