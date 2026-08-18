# KitsuMate ONNX for Unity

Unity Package Manager monorepo for KitsuMate ONNX contracts, feature packages, and selectable inference backends.

## Packages

- `ai.kitsumate.onnx` — shared contracts, model metadata, lifecycle, tokenizers, and managed dependency integration.
- `ai.kitsumate.onnx.backend.onnxruntime` — ONNX Runtime backend.
- `ai.kitsumate.onnx.backend.unity-inference` — Sentis backend (`com.unity.ai.inference` 2.6.1).
- `ai.kitsumate.onnx.{asr,embeddings,lipsync,motion,tts}` — optional feature packages.

Production packages contain no sample models, native ONNX Runtime binaries, or integration-test models. Release workflows download and validate those artifacts before publishing independent `.tgz` assets.

The ONNX Runtime release package supports Windows x64, Linux x64, and Android ARM64/ARMv7. Android uses CPU inference by default and offers opt-in NNAPI acceleration with CPU fallback.

`ExampleProject~/` is a development and verification Unity project, not the package root.

## Installation

Install the core package, one or more feature packages, and the backend package used by the application. Packages are versioned and released together; keep every `ai.kitsumate.onnx*` dependency on the same version. Test packages are optional and should be installed only in development projects.

Release assets contain independent UPM `.tgz` archives. When installing directly from Git before the first release, add every required package explicitly using a commit-pinned URL and its package subdirectory, for example:

```text
https://github.com/KitsuMate/KitsuMate.Onnx.git?path=/Packages/ai.kitsumate.onnx#<commit>
https://github.com/KitsuMate/KitsuMate.Onnx.git?path=/Packages/ai.kitsumate.onnx.backend.onnxruntime#<commit>
https://github.com/KitsuMate/KitsuMate.Onnx.git?path=/Packages/ai.kitsumate.onnx.tts#<commit>
```

Do not mix Git commits or package versions because internal UPM dependencies are coordinated atomically.

For active development inside another repository, check out this monorepo once as a Git submodule outside that project's `Assets` and `Packages` directories, then reference each required package with a local `file:` dependency. This keeps all package sources editable while the parent repository pins one exact monorepo commit. Run `Tools/ci/download-onnxruntime.py` before opening Unity because native ONNX Runtime artifacts are checksum-verified build inputs and are not stored in Git.
