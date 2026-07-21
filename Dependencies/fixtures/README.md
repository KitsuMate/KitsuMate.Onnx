# Required CPU CI fixtures

`ci.lock.json` is the single checksum-pinned external fixture set used by every Unity integration run. It is not optional and it has no smoke, release, live, or manual-selection variant.

Fixtures are downloaded into `ExampleProject~/Assets/KitsuMateOnnxFixtures` before the example project is staged for Unity. They must never be added to a UPM package archive.

Use the smallest artifact that exercises each supported feature and compatibility boundary:

- a deterministic ONNX Runtime CPU inference graph, including the minimum supported quantization;
- an ONNX model that Unity AI Inference imports and executes on CPU;
- a complete compatible Whisper set (Mel, Tiny encoder, Tiny decoder, tokenizer);
- the minimal maintained fixtures for embeddings, Uni2005 lip sync, Kimodo motion, and Chatterbox TTS.

Every entry needs a stable public URL, SHA-256, and a destination below `Assets/KitsuMateOnnxFixtures`. Missing or checksum-mismatched entries fail CI.
