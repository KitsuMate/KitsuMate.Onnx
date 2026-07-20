# KitsuMate ONNX Runtime Backend

Optional backend based on ONNX Runtime 1.24.4. Managed and native runtime artifacts are hydrated by release and CI workflows; they are intentionally not committed to this repository.

## Migrating pre-1.0 model sets

Older assets stored a concrete ONNX Runtime provider enum. After updating, run **KitsuMate > ONNX > Migration > Convert legacy ONNX Runtime compatibility**. This opt-in command updates only `Assets/` model-set assets and never runs as an import side effect.

It converts entries to the `onnxruntime` backend identifier while retaining support, memory, and operator metadata. A former provider enum cannot prove a provider-specific guarantee after conversion, so review CUDA, DirectML, TensorRT, CoreML, and NNAPI restricted models before shipping.
