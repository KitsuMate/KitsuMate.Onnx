# Changelog

## [Unreleased]

- Added provider-neutral OmniVoice ONNX inference with split and merged backbone layouts.
- Added auto voice, voice design, and transcript-required voice cloning with instruction, tag,
  pronunciation, speed, and fixed-duration controls.
- Added curated merged CPU compact INT4 and Portable FP32 profile downloads from
  `KitsuMate/omnivoice-onnx`.
- Rejected the causal upstream split INT4/CPU-FP16 exports and generated a bidirectional merged
  INT4 profile that passes Whisper checks for auto, design, and clone.
- Added deterministic prompt/model-profile tests, CPU integration coverage, and an opt-in CUDA smoke test.
- Added Chatterbox Turbo and Nano model downloads.
- Added dynamic language-model dimensions, KV caches, named graph contracts, modern text preparation, and decoder silence tokens.

## [1.0.0] - 2026-07-11

- Migrated Chatterbox to immutable engine configuration and isolated cancellable runtimes.
- Updated TextToSpeech to own a runtime while leaving the backend caller-owned.
