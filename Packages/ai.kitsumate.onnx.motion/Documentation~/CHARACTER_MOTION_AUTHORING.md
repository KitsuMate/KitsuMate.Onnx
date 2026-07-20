# Character Motion Authoring

The Character Motion authoring layer turns scene-space Humanoid targets into the semantic constraint contracts consumed by Kimodo. It does not provide runtime playback, state graphs, speech synchronization, or LLM orchestration.

## Concepts

- `CharacterMotionIntent` is a reusable prompt and optional model-identified text embedding.
- `CharacterMotion` is one concrete, avatar-specific scene motion and its baked clip.
- `CharacterMotionKeyframe` is a child GameObject at one model frame. Its transform supplies the root position and heading.
- `CharacterMotionPoseSnapshot` stores both reloadable Humanoid local rotations and compiler-ready SOMA-30 pose data.
- `CharacterMotionSkeleton` is an Editor-only transforms clone used to manipulate and capture poses. Runtime compilation never reads it.

## Setup

1. Create a `CharacterMotionIntent` from **Create > KitsuMate > Motion > Character Motion Intent** and enter the prompt.
2. Add `CharacterMotion` to an empty scene GameObject.
3. Assign the intent, target Humanoid Animator, and action origin.
4. Select **Create/Rebuild Skeleton**. The generated `MotionSkeleton` is tagged `EditorOnly` and will not enter a player build.
5. Add root, pose, or end-effector keyframes from the motion inspector.
6. For pose keyframes, select the frame, load or edit the shared skeleton, then press **Capture Pose** explicitly.
7. Create required hand/foot handles and position them with ordinary transform tools.
8. Resolve validation errors, then bake the embedding and animation.

## Coordinate and avatar rules

Unity world units are treated as metres. World positions are translated and inverse-rotated by the action origin, then reflected into Kimodo's canonical coordinate system. Action-origin scale is deliberately ignored; non-uniform, negative, or near-zero scales are rejected.

Pose snapshots include the target avatar signature and actual captured proportions. Changing the avatar invalidates captured poses and the baked clip. Optional jaw, eye, finger-end, and toe joints use documented fallbacks; missing core torso or limb bones are errors.

## Runtime use

`CharacterMotion.BuildConstraintSet()` and `BuildGenerationRequest()` are runtime-safe and work when `MotionSkeleton` is absent. Runtime code can resolve `ICharacterMotionIntent`, invoke the existing Kimodo pipeline, and consume `KimodoHumanoidMotion`. This package intentionally does not choose how that result is scheduled or played.

Runtime generation never creates project assets. Only Editor baking writes an `AnimationClip`, using avatar-specific transform paths and `AnimationUtility.SetEditorCurve`.

## Model locations

Editor baking uses the `CharacterMotionEngine`, `EmbeddingEngine`, and caller-owned backend assets assigned in the authoring inspectors. Model variants are installed and selected through model-set assets; filesystem paths and Library marker files are not supported.

Generated clips are created under `Assets/Generated/CharacterMotion` and updated in place when rebaked.
