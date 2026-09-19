# Kimodo Constraint Contracts and Controller Boundary

This package accepts constraint data, compiles it to Kimodo's normalized `[T,369]` conditioning tensors, and performs constrained ONNX inference. The authoring layer converts scene-space targets; inference uses only canonical data.

## Responsibility boundary

The authoring layer owns scene semantics: sampling transforms, selecting keyframes, path interpolation, IK/FK, retargeting, collision-aware edits, and converting Unity coordinates into the canonical model space.

The inference package owns only stable model-facing concerns:

1. Validate immutable semantic constraint records.
2. Compile canonical SOMA-30 data into Kimodo feature indices.
3. Normalize observed features using the model statistics.
4. Build the three separated-CFG branches.
5. Execute bounded ONNX windows, correct canonical poses, retain history and decode the resulting motion.

This keeps controller shapes replaceable without changing the model runtime.

## Canonical space and timing

- Space: `KimodoConstraintSpace.CanonicalYUpMeters`.
- Units: metres and radians.
- Axes: right-handed Kimodo/model coordinates, Y up.
- Timeline: integer output-frame indices in `[0, FrameCount - 1]` at 30 FPS.
- Skeleton: the public `KimodoJoint` enum is the exact SOMA-30 order.
- Heading: an XZ direction vector. The compiler normalizes it and rejects a near-zero vector.
- Arrays are frame-major. A `[N,30]` pose is passed as a flat array of `N * 30` values.

Unity-world transforms must be converted by the authoring adapter before constructing these records. The inference layer must not read `Transform`, `Animator`, `Avatar`, or scene state.

## Semantic inputs

`KimodoRootConstraint` supplies XZ root samples and optional XZ heading directions.

`KimodoFullBodyConstraint` supplies complete SOMA-30 global joint positions for each constrained frame. Kimodo derives local positions, root height, and hip-based heading from these positions. Optional smoothed root XZ samples override the default root-joint XZ values.

`KimodoEndEffectorConstraint` supplies complete SOMA-30 global positions and global rotations, then selects one or more of left hand, right hand, left foot, and right foot. Complete poses are required because NVIDIA's reference compiler expands an effector to related joints and derives root/heading features.

Multiple records can be combined in `KimodoConstraintSet`. Writing different values to the same model feature and frame is rejected as a conflict; numerically equal overlap is allowed.

## Raw model-facing input

The compiler produces a `KimodoConditioning` for the requested timeline:

- `ObservedMotion`: normalized FP32 `[T * 369]`.
- `MotionMask`: Boolean `[T * 369]`.

The object validates shape and finite observed values and copies caller arrays by default. Semantic records are preferred for application code because they prevent feature-index and normalization mistakes.

## Separated classifier-free guidance

The fixed batch-3 model receives:

| Branch | Text | Constraint mask | Observed motion |
|---|---|---|---|
| 0 | prompt | empty | repeated |
| 1 | zero | requested | repeated |
| 2 | zero | empty | repeated |

The runtime combines predictions as:

`unconditional + textGuidance * (text - unconditional) + constraintGuidance * (constraint - unconditional)`

This is intentionally not a branch where text and constraints are simultaneously enabled; it matches NVIDIA's separated-CFG pipeline. A zero constraint-guidance weight disables the constraint branch during sampling. Authored constraints still apply during canonical pose correction.

## Capability discovery

Read `CharacterMotionEngine.ConstraintCapabilities` before creating constraints. The current v1.1 graph supports 2–300 frames per inference window, 30 FPS, 369 features, 30 internal joints, a 4096-value text embedding, separated guidance, root position/heading, full-body pose, and all four supported end effectors.

Controllers should fail or adapt at their boundary when a capability is unavailable. They should not infer support from a model filename.

## High-level invocation

Use `Llm2VecEmbeddingEngine` for CPU text encoding and `CharacterMotionEngineRuntime` for semantic compilation and motion generation. Frequently reused prompts should store their embedding in `CharacterMotionIntent`; prebaked intents do not load the embedding runtime.

## Validation

Compiler fixtures compare masks and observed features with NVIDIA's original classes. Runtime tests cover variable lengths, multi-window duration, timeline constraints, prompt changes, previous history and cancellation. Optional model integration tests reuse a session across lengths and exercise long CUDA sequences.

Pose correction preserves canonical bone lengths and fits feasible targets. It is a managed implementation, not NVIDIA's native solver; exact native postprocessing parity is not claimed. See the package README for the sequence behavior and model source.
