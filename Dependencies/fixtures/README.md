# Required CPU CI fixtures

`ci.lock.json` is the single checksum-pinned external fixture set used by every Unity integration run. It is not optional and it has no smoke, release, live, or manual-selection variant.

The example project is staged first. Fixtures are then downloaded into `ExampleProject/KitsuMateOnnxFixtures`; the Unity AI Inference fixture is downloaded into `ExampleProject/Assets/KitsuMateOnnxFixtures` so Unity imports it as a `ModelAsset`. They must never be added to a UPM package archive.

Use the smallest artifact that exercises each supported feature and compatibility boundary:

- a deterministic ONNX Runtime CPU inference graph, including the minimum supported quantization;
- an ONNX model that Unity AI Inference imports and executes on CPU;
- a complete compatible Whisper set (Mel, Tiny encoder, Tiny decoder, tokenizer);
- the minimal maintained fixtures for embeddings, Uni2005 lip sync, Kimodo motion, and Chatterbox TTS.

Every entry needs an immutable public URL, SHA-256, and an explicit destination. Missing or checksum-mismatched entries fail CI.
