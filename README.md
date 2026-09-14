# KitsuMate ONNX for Unity

Unity Package Manager monorepo for KitsuMate ONNX contracts, feature packages, and selectable inference backends.

## Packages

- `ai.kitsumate.onnx` — shared contracts, model metadata, lifecycle, tokenizers, and managed dependency integration.
- `ai.kitsumate.onnx.backend.onnxruntime` — ONNX Runtime backend.
- `ai.kitsumate.onnx.backend.unity-inference` — Sentis backend (`com.unity.ai.inference` 2.6.1).
- `ai.kitsumate.onnx.{asr,embeddings,lipsync,motion,tts}` — optional feature packages.

The default ONNX Runtime package includes its managed binding, native libraries, licenses, and Unity import settings in ordinary Git. Git dependencies, local submodules, and released `.tgz` archives use the same checksum-pinned files; no consumer downloader or Git LFS setup is required. Model weights and integration-test models are not included.

The default backend selects DirectML on Windows x64, WebGPU on Linux x64, CoreML on Apple-silicon macOS, and NNAPI on Android ARM64/ARMv7, with CPU fallback.

`ExampleProject~/` is a development and verification Unity project, not the package root.

## Installation

Install the core package, one or more feature packages, and the backend package used by the application. Packages are versioned and released together; keep every `ai.kitsumate.onnx*` dependency on the same version. Test packages are optional and should be installed only in development projects.

Release assets contain independent UPM `.tgz` archives. When installing directly from Git, add every required package explicitly using a commit-pinned URL and its package subdirectory, for example:

```text
https://github.com/KitsuMate/KitsuMate.Onnx.git?path=/Packages/ai.kitsumate.onnx#<commit>
https://github.com/KitsuMate/KitsuMate.Onnx.git?path=/Packages/ai.kitsumate.onnx.backend.onnxruntime#<commit>
https://github.com/KitsuMate/KitsuMate.Onnx.git?path=/Packages/ai.kitsumate.onnx.tts#<commit>
```

Do not mix Git commits or package versions because internal UPM dependencies are coordinated atomically.

For active development inside another repository, check out this monorepo once as a Git submodule outside that project's `Assets` and `Packages` directories, then reference each required package with a local `file:` dependency. This keeps all package sources editable while the parent repository pins one exact monorepo commit. Open Unity directly after checkout; the default backend is complete.

The large optional `ai.kitsumate.onnx.backend.onnxruntime.nvidia` package is distributed as a complete release `.tgz`. Install that archive when enabling NVIDIA profiles. Its Git source alone does not include the CUDA/TensorRT dependencies. Default profiles do not require this optional package.

Maintainers validate runtime updates offline before committing the binaries and lock together:

```powershell
python Tools/ci/validate-onnxruntime.py --lock Dependencies/onnxruntime.lock.json --directory Packages/ai.kitsumate.onnx.backend.onnxruntime/Runtime/Plugins
```

CI checks every bundled file against `Dependencies/onnxruntime.lock.json`. Releases
package the same validated files without downloading another default runtime.
