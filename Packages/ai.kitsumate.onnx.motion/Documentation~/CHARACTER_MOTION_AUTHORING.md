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
4. Choose a frame in the motion inspector and select **Add Pose**. Existing frames change the action to **Select Pose**.
5. Select the keyframe and enable root, full-pose, or hand/foot constraints. The shared `MotionSkeleton` and targets are created automatically. Poles are created only for the deterministic Full Pose preview.
6. Rotate bones or manipulate hand/foot IK in the Scene view. Use Unity's Move tool for targets and poles and the Rotate tool for target orientation. Pose changes are captured automatically, including edits made through the ordinary Transform inspector or tools.
7. Use **Reset Pose** or **Sample Pose** for deliberate pose replacement. Use **Recalculate** on the motion inspector to repair missing or stale pose data in one batch.
8. Resolve validation errors, then bake the embedding and animation.

Full Pose armatures remain visible when their keyframes are not selected. Without Full Pose, the armature and poles are hidden: Kimodo treats the remaining targets as sparse generative constraints and may reposition the whole body to satisfy them. These targets have no preview limb-length restriction. Bone and IK geometry scales with the complete reference armature.

IK targets are drawn as wireframe boxes. Elbow and knee poles are drawn as camera-facing wireframe circles with a line to the middle bone they control.

Unselected roots, Full Pose bones and poles, and visible IK boxes are explicit Scene-view controls. Clicking one selects its keyframe and carries the clicked subcontrol into the selected editor; real Unity move and rotation handles retain input priority.

The `CharacterMotionKeyframe` Transform is the only authoring source for root position and rotation. Pose snapshots are stored relative to it, the shared guide is rebased to it whenever a frame is loaded, and keyframe scale must remain one. Snapshots created before this root-space contract are reported by **Recalculate** and migrated in the batch operation.

## Coordinate and avatar rules

Unity world units are treated as metres. World positions are translated and inverse-rotated by the action origin, then reflected into Kimodo's canonical coordinate system. Action-origin scale is deliberately ignored; non-uniform, negative, or near-zero scales are rejected.

Pose snapshots include the target avatar signature and actual captured proportions. Changing the avatar invalidates captured poses and the baked clip. Optional jaw, eye, finger-end, and toe joints use documented fallbacks; missing core torso or limb bones are errors.

## Runtime use

`CharacterMotion.BuildConstraintSet()` and `BuildGenerationRequest()` are runtime-safe and work when `MotionSkeleton` and editor-only pole controls are absent. Runtime code can resolve `ICharacterMotionIntent`, invoke the existing Kimodo pipeline, and consume `KimodoHumanoidMotion`. This package intentionally does not choose how that result is scheduled or played.

Runtime generation never creates project assets. Only Editor baking writes an `AnimationClip`, using avatar-specific transform paths and `AnimationUtility.SetEditorCurve`.

## Model locations

Editor baking uses the `CharacterMotionEngine`, `EmbeddingEngine`, and caller-owned backend assets assigned in the authoring inspectors. Model variants are installed and selected through model-set assets; filesystem paths and Library marker files are not supported.

Generated clips are created under `Assets/Generated/CharacterMotion` and updated in place when rebaked.
