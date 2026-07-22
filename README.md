# KitsuMate ONNX for Unity

Unity Package Manager monorepo for KitsuMate ONNX contracts, feature packages, and selectable inference backends.

## Packages

- `ai.kitsumate.onnx` — shared contracts, model metadata, lifecycle, tokenizers, and managed dependency integration.
- `ai.kitsumate.onnx.backend.onnxruntime` — ONNX Runtime backend.
- `ai.kitsumate.onnx.backend.sentis` — Unity AI Inference backend (`com.unity.ai.inference`).
- `ai.kitsumate.onnx.{asr,embeddings,lipsync,motion,tts}` — optional feature packages.

Production packages contain no sample models, native ONNX Runtime binaries, or integration-test models. Release workflows download and validate those artifacts before publishing independent `.tgz` assets.

`ExampleProject~/` is a development and verification Unity project, not the package root.
