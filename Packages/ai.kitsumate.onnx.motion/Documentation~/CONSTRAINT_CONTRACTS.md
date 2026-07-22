# Kimodo Constraint Contracts and Controller Boundary

This package accepts constraint data, compiles it to Kimodo's normalized `[60,369]` conditioning tensors, and performs constrained ONNX inference. It intentionally does not define pose-design, path-authoring, IK, scene-query, or `MonoBehaviour` controller APIs yet.

## Responsibility boundary

Future controllers own authoring and scene semantics: sampling transforms, selecting keyframes, path interpolation, IK/FK, retargeting, collision-aware edits, and converting Unity coordinates into the canonical model space.

The inference package owns only stable model-facing concerns:

1. Validate immutable semantic constraint records.
2. Compile canonical SOMA-30 data into Kimodo feature indices.
3. Normalize observed features using the model statistics.
4. Build the three separated-CFG branches.
5. Execute ONNX inference and decode the resulting motion.

This keeps controller shapes replaceable without changing the model runtime.

## Canonical space and timing

- Space: `KimodoConstraintSpace.CanonicalYUpMeters`.
- Units: metres and radians.
- Axes: right-handed Kimodo/model coordinates, Y up.
- Timeline: integer model-frame indices in `[0,59]` at 30 FPS.
- Skeleton: the public `KimodoJoint` enum is the exact SOMA-30 order.
- Heading: an XZ direction vector. The compiler normalizes it and rejects a near-zero vector.
- Arrays are frame-major. A `[N,30]` pose is passed as a flat array of `N * 30` values.

Unity-world transforms must be converted by a future adapter before constructing these records. The inference layer must not read `Transform`, `Animator`, `Avatar`, or scene state.

## Semantic inputs

`KimodoRootConstraint` supplies XZ root samples and optional XZ heading directions.

`KimodoFullBodyConstraint` supplies complete SOMA-30 global joint positions for each constrained frame. Kimodo derives local positions, root height, and hip-based heading from these positions. Optional smoothed root XZ samples override the default root-joint XZ values.

`KimodoEndEffectorConstraint` supplies complete SOMA-30 global positions and global rotations, then selects one or more of left hand, right hand, left foot, and right foot. Complete poses are required because NVIDIA's reference compiler expands an effector to related joints and derives root/heading features.

Multiple records can be combined in `KimodoConstraintSet`. Writing different values to the same model feature and frame is rejected as a conflict; numerically equal overlap is allowed.

## Raw model-facing input

Advanced callers and tests can pass a compiled `KimodoConditioning` directly:

- `ObservedMotion`: normalized FP32 `[60 * 369]`.
- `MotionMask`: Boolean `[60 * 369]`.

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

This is intentionally not a branch where text and constraints are simultaneously enabled; it matches NVIDIA's separated-CFG pipeline. With an empty mask, the compiled path is exactly regression-tested against the original prompt-only overload. With `constraintGuidance = 0`, constraints do not alter the result.

## Capability discovery

Read `CharacterMotionEngine.ConstraintCapabilities` before creating constraints. The current v1.1 graph reports fixed 60 frames, 30 FPS, 369 features, 30 internal joints, a 4096-value text embedding, separated guidance, root position/heading, full-body pose, and all four supported end effectors.

Controllers should fail or adapt at their boundary when a capability is unavailable. They should not infer support from a model filename.

## High-level invocation

Use `Llm2VecEmbeddingEngine` for CPU text encoding and `CharacterMotionEngineRuntime` for semantic compilation and motion generation. Frequently reused prompts should store their embedding in `CharacterMotionIntent`; prebaked intents do not load the embedding runtime.

## Validation gates

- Python generates root, full-body, and end-effector observed-motion/mask fixtures using NVIDIA's original classes.
- Unity compiler parity requires exact masks and tight FP32 value agreement.
- CFG branch construction has an exact unit test.
- Empty conditioning and zero constraint guidance have exact output regressions on the real CUDA model.
- Every semantic constraint type is run through the real model and checked for finite, changed output.
- A longer root-conditioned run must reduce normalized target error relative to an identical unconstrained run.
- The public prompt pipeline is tested as CPU text encoding → semantic root compilation → CUDA motion inference → finite Humanoid motion.

## Deferred controller work

Later packages may add root-path, target-pose, inbetweening, or end-effector authoring components. Those components should depend on these semantic records, not on ONNX tensors. Their design should be driven by actual authoring workflows and can add interpolation, feasibility checks, FK/IK, retargeting, and visualization without expanding the inference contract.
