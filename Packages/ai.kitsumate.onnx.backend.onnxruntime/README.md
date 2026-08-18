# KitsuMate ONNX Runtime Backend

Optional backend based on ONNX Runtime 1.24.4. Managed and native runtime artifacts are downloaded by release and CI workflows; they are intentionally not committed to this repository.

The published package supports Windows x64, Linux x64, and Android ARM64/ARMv7. The default provider preference is `TensorRt, Cuda, DirectMl, OpenVino, Nnapi, Cpu`. Providers that are invalid for the current platform or absent from the active Build Profile are skipped without changing the configured order. CPU therefore remains the portable fallback without preventing an included accelerator from being preferred.

On Android the first use of each external model copies it from the APK's read-only `StreamingAssets/KitsuMateModels` directory to `Application.persistentDataPath/KitsuMateModels`, validates its required SHA-256, and opens the staged filesystem path. The native runtime requires API 24, while Unity 6000.5 requires projects to target API 26 or newer. Use `SetProviderOrder(OnnxExecutionProvider.Nnapi, OnnxExecutionProvider.Cpu)` for NNAPI with initialization fallback, or omit `Cpu` to require NNAPI initialization. NNAPI requires API 27. ONNX Runtime can still assign unsupported graph nodes to its CPU implementation even when CPU is omitted from the initialization policy.

Providers are selected by the backend's ordered `ProviderOrder` and must also be included by the active Unity Build Profile. Supported defines are `KITSUMATE_ORT_DIRECTML`, `KITSUMATE_ORT_CUDA`, `KITSUMATE_ORT_TENSORRT`, `KITSUMATE_ORT_OPENVINO`, and Android-only `KITSUMATE_ORT_NNAPI`. TensorRT requires CUDA; DirectML is Windows-only; OpenVINO is Linux-only. An empty list and duplicate entries are invalid. Use `[Cpu]` for CPU-only operation or omit `Cpu` when initialization must fail instead of falling back.

DirectML and OpenVINO payloads are not published until the pinned ONNX Runtime 1.24.4 combined-core feasibility gate passes. Do not hydrate them from separate official packages: the core and every provider library must come from the same source build. The pre-build validator intentionally rejects a profile that enables a provider whose payload is absent.

DirectML, macOS, and iOS native runtimes are not bundled.
