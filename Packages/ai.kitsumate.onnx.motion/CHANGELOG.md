# Changelog

## [1.0.0] - 2026-07-11

- Added CharacterMotionEngine, KimodoEngine, and KimodoModelSet.
- Routed authoring through selected motion and embedding engines/backends.
- Removed public raw-path Kimodo generator, text encoder, pipeline, environment variables, and marker files.

All notable changes to this project are documented here.

## [0.3.0] - 2026-07-11

### Added

- `CharacterMotionIntent` and runtime `ICharacterMotionIntent` with model-identified embedding caching.
- Scene `CharacterMotion` and `CharacterMotionKeyframe` authoring contracts.
- Deterministic avatar-specific Humanoid/SOMA pose snapshots and validation diagnostics.
- Tested Humanoid-to-SOMA mapping with explicit optional-bone fallbacks.
- Editor-only full-hierarchy guide skeleton, pose capture, animation sampling, gizmos, and effector handles.
- CPU embedding and CUDA motion bake commands producing avatar-specific Unity `AnimationClip` assets.
- Character Motion authoring documentation and unit/project integration coverage.

## [0.2.0] - 2026-07-10

### Added

- First Kimodo SOMA RP v1.1 end-to-end invocation service.
- Pre-generated LLM2Vec embedding contract.
- Fixed 60-frame separated-CFG/DDIM sampler.
- SOMA feature decoding into Unity Humanoid rotation deltas and root motion.
- External model-path loading and model integration tests.
- Independent CPU LLM2Vec ONNX text encoder, exact Llama 3 tokenization, and prompt-to-motion pipeline.
- Python/C# token fixtures plus standard-BF16 and NF4 embedding reference fixtures.
- Large-model CPU text-encoder integration is now selected through an LLM2Vec engine/model-set fixture.
- Real CPU-text-to-CUDA-motion large-model integration smoke test.
- Public capability discovery plus raw `[60,369]` conditioning input.
- Canonical SOMA-30 root, full-body, and end-effector constraint contracts and compiler.
- Separated text/constraint CFG with independent guidance weights.
- Python constraint fixtures, Unity parity tests, empty-path regressions, and real CUDA constraint/adherence tests.
- High-level constrained CPU-text-to-CUDA-motion pipeline overloads.

### Changed

- Replaced ScriptableObject-owned engine state with a plain disposable runtime service.
- Motion output now uses flat Quaternion/Vector3 arrays instead of Euler multidimensional arrays.

### Removed

- Early EMAGE audio-driven implementation.

## [0.1.0] - 2026-01-05

### Added

- Initial experimental EMAGE package.
