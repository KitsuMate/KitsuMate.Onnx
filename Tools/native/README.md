# ONNX Runtime provisioning

The default backend ships ONNX Runtime 1.30.0 for Windows, Linux, macOS, and Android,
with matching managed bindings. Windows and Linux use the WebGPU 0.3.0 plugin.
Provider packages have independent version numbers from the core runtime.

`Dependencies/onnxruntime.lock.json` pins official NuGet artifacts and each installed
file's SHA-256. The complete default payload and Unity-generated importer metadata
are tracked as ordinary Git files, so local, Git UPM, and release archive consumers
use the same files without an application-specific download or copy step.

Run from this repository to provision a reviewed lock:

```powershell
python Tools/native/update-onnxruntime.py --lock Dependencies/onnxruntime.lock.json --cache .cache/onnxruntime --destination Packages/ai.kitsumate.onnx.backend.onnxruntime/Runtime/Plugins
```

The updater stages and verifies downloads before replacing installed files. Close
Unity before replacing loaded native DLLs. After an update, open Unity and run
`OnnxRuntimePluginSetup.ConfigureWindowsImporters()` and `ConfigureAndroidImporters()`
when adding native files; commit the generated metadata with the payload. Run
`Tools/ci/validate-package-layout.py` to verify package contents and checksums.

The optional NVIDIA package uses the same updater with `onnxruntime-nvidia.lock.json`.
Its ORT providers match the core version; CUDA, cuDNN and TensorRT-RTX use their own
version numbers. Release CI provisions that package before assembling its archive.
