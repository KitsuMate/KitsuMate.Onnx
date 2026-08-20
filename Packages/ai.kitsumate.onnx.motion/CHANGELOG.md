# Changelog

## Unreleased

- Made explicit root position and heading authoritative over pose-derived context at the same frame, preventing valid root-plus-IK/full-pose combinations from reporting feature conflicts.
- Reused the compiled constraint set when building an engine request instead of validating and compiling it twice.
- Replaced the raw keyframe inspector with constraint toggles, automatic guide/pose maintenance, Reset Pose, and frame-based animation sampling.
- Added compact motion-level pose insertion and batch recalculation for missing or stale keyframes.
- Added SOMA-30 Scene handles with capsule bones, box IK controls, FK rotation, pole-guided two-bone IK, Undo, and automatic pose capture.
- Scaled bone and IK geometry from the armature extent, separated target/pole picking, and made IK manipulation follow Unity's Move and Rotate tools.
- Kept captured poses visible while their keyframes are unselected and synchronized IK edits made with ordinary Transform tools.
- Rebased reset, load, sampling, IK creation, and selected gizmos to the owning keyframe Transform so shared-guide state cannot leak between frames.
- Added automatic stale detection for legacy guide-root snapshots and rejected keyframe scale because root authoring consists only of position and rotation.
- Changed IK targets to wireframe boxes and poles to wireframe circles connected to their elbow or knee.
- Oriented IK boxes with their hand or foot, centered and sized them around the corresponding SOMA joints while retaining the model-input point as their manipulation pivot, and made selected controls almost white.
- Gave active transform and rotation handles input priority over overlapping armature and IK selection regions.
- Treated IK-only keyframes as sparse generative constraints: their bones and poles stay hidden, targets remain unrestricted, and local two-bone solving is reserved for Full Pose.
- Replaced unreliable pickable gizmos with cached explicit Scene controls that select unselected keyframe roots, bones, IK targets, and poles while preserving Unity handle priority.

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
