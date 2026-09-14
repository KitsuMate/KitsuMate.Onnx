# ONNX Runtime native builds

The default Windows DirectML core and NVIDIA CUDA plug-in are built from the
exact ONNX Runtime v1.25.1 commit `8a77e459420f58fb946fd9067285cfa719f10bdd`.
CUDA builds require CUDA 12.9 and cuDNN 9. Release artifacts are promoted only
after PE/ELF dependency inspection and their output hashes are added to the
corresponding artifact lock.

The `Build ONNX Runtime DirectML artifact` workflow produces a maintainer archive
and SHA-256 list. Review the output, update `Dependencies/onnxruntime.lock.json`,
and commit the default backend DLLs and Unity-generated importer metadata together.
Consumers use those tracked files directly; there is no separate native release
or runtime downloader in the default installation path.

`update-onnxruntime.py` is a maintainer tool. It stages and verifies all selected
files before changing the installed payload. Failed downloads or checksum checks
leave the existing files intact. For source-built Windows DLLs, pass the reviewed
build output directory:

```powershell
python Tools/native/update-onnxruntime.py --lock Dependencies/onnxruntime.lock.json --cache .cache/onnxruntime --destination Packages/ai.kitsumate.onnx.backend.onnxruntime/Runtime/Plugins --platform Windows --source-build <build-output>
```

For a full default-runtime update, omit `--platform`. Preserve all platform DLLs,
libraries, licenses, and checksums in the same change. Run
`Tools/ci/validate-package-layout.py` before committing. Do not use Git LFS for
this payload; ordinary Git files keep UPM Git installation self-contained.

TensorRT-RTX is not built here. The NVIDIA package consumes checksum-pinned
standalone EP ABI v0.3.0/cu12 release artifacts.
