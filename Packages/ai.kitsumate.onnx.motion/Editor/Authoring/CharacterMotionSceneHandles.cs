using System;
using System.Collections.Generic;
using KitsuMate.Onnx.Motion.Kimodo;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace KitsuMate.Onnx.Motion.Editor
{
    internal sealed class CharacterMotionSceneHandles
    {
        internal static readonly Color Muted = new Color(0.38f, 0.4f, 0.42f, 1f);
        internal static readonly Color Enabled = new Color(0.12f, 0.72f, 0.7f, 1f);
        internal static readonly Color Invalid = new Color(1f, 0.48f, 0.12f, 1f);
        internal static readonly Color Selected = new Color(0.94f, 0.97f, 0.97f, 1f);

        internal static readonly (int a, int b)[] Segments =
        {
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (6, 8), (6, 9),
            (3, 10), (10, 11), (11, 12), (12, 13), (13, 14), (13, 15),
            (3, 16), (16, 17), (17, 18), (18, 19), (19, 20), (19, 21),
            (0, 22), (22, 23), (23, 24), (24, 25),
            (0, 26), (26, 27), (27, 28), (28, 29),
        };

        private static readonly HumanBodyBones?[] Bones =
        {
            HumanBodyBones.Hips, HumanBodyBones.Spine, HumanBodyBones.Chest, HumanBodyBones.UpperChest,
            null, HumanBodyBones.Neck, HumanBodyBones.Head, HumanBodyBones.Jaw,
            HumanBodyBones.LeftEye, HumanBodyBones.RightEye,
            HumanBodyBones.LeftShoulder, HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand,
            null, null,
            HumanBodyBones.RightShoulder, HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand,
            null, null,
            HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot, HumanBodyBones.LeftToes,
            HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot, HumanBodyBones.RightToes,
        };

        internal enum IkPart
        {
            None,
            Target,
            Pole,
        }

        internal readonly struct IkPick
        {
            internal readonly CharacterMotionConstraintType Type;
            internal readonly IkPart Part;
            internal bool IsValid => Type != CharacterMotionConstraintType.None && Part != IkPart.None;

            internal IkPick(CharacterMotionConstraintType type, IkPart part)
            {
                Type = type;
                Part = part;
            }
        }

        internal readonly struct IkBoxGeometry
        {
            internal readonly Vector3 Pivot;
            internal readonly Vector3 Center;
            internal readonly Quaternion Rotation;
            internal readonly Vector3 Size;

            internal IkBoxGeometry(Vector3 pivot, Vector3 center, Quaternion rotation, Vector3 size)
            {
                Pivot = pivot;
                Center = center;
                Rotation = rotation;
                Size = size;
            }
        }

        private readonly CharacterMotionKeyframe keyframe;
        private readonly Vector3[] positions = new Vector3[30];
        private readonly Quaternion[] rotations = new Quaternion[30];
        private readonly CharacterMotionValidationResult diagnostics = new CharacterMotionValidationResult();
        private int selectedJoint = -1;
        private IkPick selectedIk;
        private bool poseEditActive;
        private int poseEditUndoGroup = -1;
        private CharacterMotionSkeleton pendingSkeleton;
        private string pendingOperation;

        internal CharacterMotionSceneHandles(CharacterMotionKeyframe value)
        {
            keyframe = value;
            CharacterMotionUnselectedSceneControls.ConsumeSelection(value,
                out selectedJoint, out selectedIk);
        }

        internal void OnSceneGUI()
        {
            if (keyframe == null) return;
            CharacterMotion motion = keyframe.GetComponentInParent<CharacterMotion>();
            if (motion == null) return;
            bool fullPose = CharacterMotionAuthoringUtility.HasFullPose(keyframe);
            CharacterMotionSkeleton skeleton = null;
            bool validPose;
            if (fullPose)
            {
                if (motion.TargetAnimator == null) return;
                try { skeleton = CharacterMotionAuthoringUtility.EnsureSkeleton(motion); }
                catch (Exception) { return; }
                if (skeleton.GuideAnimator == null) return;
                // Moving or rotating the keyframe rebases the pose; it must not leave the
                // shared guide behind at its build position or at another frame's root.
                if (skeleton.LoadedKeyframe != keyframe)
                {
                    if (keyframe.Pose.IsCaptured) skeleton.LoadPose(keyframe);
                    else skeleton.LoadBindPose(keyframe);
                }
                else
                    skeleton.AlignToKeyframe(keyframe);

                diagnostics.Clear();
                validPose = HumanoidSomaMapper.TryCapture(skeleton.GuideAnimator.GetBoneTransform,
                    positions, rotations, diagnostics);
            }
            else
                validPose = TryReconstructPose(keyframe, positions, rotations);

            float armatureSize = MeasureArmature(positions);
            int hoveredJoint = fullPose ? FindHoveredBone(positions) : -1;
            IkPick hoveredIk = hoveredJoint < 0 ? FindHoveredIk(positions, rotations, armatureSize, fullPose) : default;

            DrawRoot(keyframe, armatureSize);
            if (fullPose) DrawBones(positions, validPose, hoveredJoint, armatureSize);
            DrawIk(hoveredIk, skeleton, armatureSize, fullPose);
            // Unity's active transform handle must see MouseDown first. In particular,
            // rotation rings often overlap the rendered bones and IK bounds; allowing
            // custom picking first would replace the selection instead of starting a drag.
            DrawManipulators(skeleton);
            HandleSelection(hoveredJoint, hoveredIk);
            CompletePoseEditIfReleased();
        }

        internal static void DrawUnselectedKeyframe(
            CharacterMotionKeyframe value,
            Vector3[] positions,
            Quaternion[] rotations,
            bool valid)
        {
            float armatureSize = MeasureArmature(positions);
            DrawRoot(value, armatureSize);
            if (CharacterMotionAuthoringUtility.HasFullPose(value))
            {
                foreach ((int a, int b) in Segments)
                {
                    Color color = valid
                        ? IsJointEnabled(value.Constraints, b) ? Enabled : Muted
                        : Invalid;
                    DrawCapsule(positions[a], positions[b], color, BoneRadius(armatureSize));
                }
            }
            DrawStoredIk(value, positions, rotations, armatureSize);
        }

        internal static float MeasureArmature(Vector3[] positions)
        {
            if (positions == null || positions.Length == 0) return 1f;
            int firstFinite = Array.FindIndex(positions, CharacterMotionSpace.IsFinite);
            if (firstFinite < 0) return 1f;
            var bounds = new Bounds(positions[firstFinite], Vector3.zero);
            for (int i = firstFinite + 1; i < positions.Length; i++)
            {
                if (CharacterMotionSpace.IsFinite(positions[i])) bounds.Encapsulate(positions[i]);
            }
            return Mathf.Max(0.1f, bounds.size.magnitude);
        }

        internal static float BoneRadius(float armatureSize) => Mathf.Max(0.001f, armatureSize * 0.0045f);
        internal static float IkSize(float armatureSize) => Mathf.Max(0.01f, armatureSize * 0.04f);

        internal static bool TryReconstructPose(
            CharacterMotionKeyframe value,
            Vector3[] targetPositions,
            Quaternion[] targetRotations)
        {
            ReadOnlySpan<Vector3> relativePositions = value.Pose.SomaPositionsRelativeToRoot.Span;
            ReadOnlySpan<Quaternion> relativeRotations = value.Pose.SomaRotationsRelativeToRoot.Span;
            bool valid = value.Pose.IsCaptured && relativePositions.Length == targetPositions.Length &&
                         relativeRotations.Length == targetRotations.Length;
            for (int i = 0; i < targetPositions.Length; i++)
            {
                Vector3 position = i < relativePositions.Length ? relativePositions[i] : Vector3.zero;
                Quaternion rotation = i < relativeRotations.Length ? relativeRotations[i] : Quaternion.identity;
                valid &= CharacterMotionSpace.IsFinite(position) && CharacterMotionSpace.IsFinite(rotation);
                targetPositions[i] = value.transform.position + value.transform.rotation * position;
                targetRotations[i] = value.transform.rotation * rotation;
            }
            return valid;
        }

        private int FindHoveredBone(Vector3[] positions)
        {
            float best = 9f;
            int result = -1;
            foreach ((int a, int b) in Segments)
            {
                float distance = HandleUtility.DistanceToLine(positions[a], positions[b]);
                if (distance >= best) continue;
                best = distance;
                result = b;
            }
            return result;
        }

        private IkPick FindHoveredIk(
            Vector3[] somaPositions,
            Quaternion[] somaRotations,
            float armatureSize,
            bool includePoles)
        {
            float best = 10f;
            IkPick result = default;
            foreach (CharacterMotionConstraintType type in CharacterMotionAuthoringUtility.EffectorTypes)
            {
                if ((keyframe.Constraints & type) == 0) continue;
                Transform target = keyframe.GetEffector(type);
                Transform pole = CharacterMotionAuthoringUtility.FindPole(keyframe, type);
                if (target != null)
                {
                    IkBoxGeometry box = CalculateIkBox(type, target.position, target.rotation,
                        somaPositions, somaRotations, armatureSize);
                    Check(DistanceToBox(box), new IkPick(type, IkPart.Target));
                }
                if (includePoles && pole != null)
                    Check(HandleUtility.DistanceToCircle(pole.position, IkSize(armatureSize) * 0.39f),
                        new IkPick(type, IkPart.Pole));
            }
            return result;

            void Check(float distance, IkPick pick)
            {
                if (distance >= best) return;
                best = distance;
                result = pick;
            }
        }

        private void HandleSelection(int hoveredJoint, IkPick hoveredIk)
        {
            Event current = Event.current;
            if (current.type != EventType.MouseDown || current.button != 0 || current.alt) return;
            if (hoveredJoint >= 0)
            {
                selectedJoint = hoveredJoint;
                selectedIk = default;
                current.Use();
            }
            else if (hoveredIk.IsValid)
            {
                selectedIk = hoveredIk;
                selectedJoint = -1;
                current.Use();
            }
        }

        private static void DrawRoot(CharacterMotionKeyframe value, float armatureSize)
        {
            bool positionEnabled = (value.Constraints & CharacterMotionConstraintType.RootPosition) != 0;
            bool headingEnabled = (value.Constraints & CharacterMotionConstraintType.RootHeading) != 0;
            float size = armatureSize * 0.025f;
            Handles.color = positionEnabled ? Enabled : Muted;
            Handles.SphereHandleCap(0, value.transform.position, Quaternion.identity, size, EventType.Repaint);
            Handles.color = headingEnabled ? Enabled : Muted;
            Handles.DrawAAPolyLine(3f, value.transform.position,
                value.transform.position + value.transform.forward * armatureSize * 0.12f);
        }

        private void DrawBones(Vector3[] positions, bool validPose, int hoveredJoint, float armatureSize)
        {
            foreach ((int a, int b) in Segments)
            {
                bool enabled = IsJointEnabled(keyframe.Constraints, b);
                Color color = validPose ? enabled ? Enabled : Muted : Invalid;
                if (hoveredJoint == b || selectedJoint == b) color = Lighten(color);
                DrawCapsule(positions[a], positions[b], color, BoneRadius(armatureSize));
            }
            int labelJoint = hoveredJoint >= 0 ? hoveredJoint : selectedJoint;
            if (labelJoint >= 0)
            {
                Handles.color = Lighten(IsJointEnabled(keyframe.Constraints, labelJoint) ? Enabled : Muted);
                Handles.Label(positions[labelJoint], ((KimodoJoint)labelJoint).ToString(), EditorStyles.boldLabel);
            }
        }

        private void DrawIk(
            IkPick hovered,
            CharacterMotionSkeleton skeleton,
            float armatureSize,
            bool fullPose)
        {
            float size = IkSize(armatureSize);
            foreach (CharacterMotionConstraintType type in CharacterMotionAuthoringUtility.EffectorTypes)
            {
                bool enabled = (keyframe.Constraints & type) != 0;
                Transform target = keyframe.GetEffector(type);
                Transform pole = CharacterMotionAuthoringUtility.FindPole(keyframe, type);
                bool invalid = enabled && (target == null ||
                    fullPose && (pole == null || IsUnreachable(type, target, skeleton)));
                Color baseColor = invalid ? Invalid : enabled ? Enabled : Muted;
                Color targetColor = IsCurrent(hovered, type, IkPart.Target) || IsCurrent(selectedIk, type, IkPart.Target)
                    ? Selected : baseColor;
                Color poleColor = IsCurrent(hovered, type, IkPart.Pole) || IsCurrent(selectedIk, type, IkPart.Pole)
                    ? Selected : baseColor;
                if (target != null)
                {
                    IkBoxGeometry box = CalculateIkBox(type, target.position, target.rotation,
                        positions, rotations, armatureSize);
                    DrawBox(box, targetColor);
                }
                Transform middle = null;
                if (fullPose)
                    CharacterMotionAuthoringUtility.GetChain(skeleton.GuideAnimator, type,
                        out _, out middle, out _);
                if (fullPose && pole != null && middle != null)
                    DrawPole(pole.position, middle.position, poleColor, size * 0.39f);

                IkPick label = hovered.IsValid ? hovered : selectedIk;
                if (label.Type == type)
                {
                    Transform control = label.Part == IkPart.Target ? target : pole;
                    if (control != null)
                    {
                        Handles.color = label.Part == IkPart.Target ? targetColor : poleColor;
                        Handles.Label(control.position, ControlLabel(type, label.Part), EditorStyles.boldLabel);
                    }
                }
            }
        }

        private void DrawManipulators(CharacterMotionSkeleton skeleton)
        {
            bool fullPose = CharacterMotionAuthoringUtility.HasFullPose(keyframe);
            if (fullPose && selectedJoint >= 0 && selectedJoint < Bones.Length && Bones[selectedJoint].HasValue)
            {
                Transform bone = skeleton.GuideAnimator.GetBoneTransform(Bones[selectedJoint].Value);
                if (bone != null)
                {
                    EditorGUI.BeginChangeCheck();
                    Quaternion boneRotation = Handles.RotationHandle(bone.rotation, bone.position);
                    if (EditorGUI.EndChangeCheck())
                    {
                        BeginPoseEdit(skeleton, "Rotate Character Motion Bone", bone);
                        bone.rotation = boneRotation;
                    }
                }
            }

            if (!selectedIk.IsValid || (keyframe.Constraints & selectedIk.Type) == 0) return;
            Transform target = keyframe.GetEffector(selectedIk.Type);
            Transform pole = CharacterMotionAuthoringUtility.FindPole(keyframe, selectedIk.Type);
            if (!fullPose && selectedIk.Part == IkPart.Pole) return;
            Transform control = selectedIk.Part == IkPart.Target ? target : pole;
            if (control == null) return;

            EditorGUI.BeginChangeCheck();
            Vector3 position = control.position;
            Quaternion rotation = control.rotation;
            if (selectedIk.Part == IkPart.Pole || Tools.current is Tool.Move or Tool.Rect or Tool.Transform)
                position = Handles.PositionHandle(control.position,
                    selectedIk.Part == IkPart.Target ? control.rotation : Quaternion.identity);
            else if (selectedIk.Part == IkPart.Target && Tools.current == Tool.Rotate)
                rotation = Handles.RotationHandle(control.rotation, control.position);
            else
                return;
            if (!EditorGUI.EndChangeCheck()) return;

            if (!fullPose)
            {
                Undo.RecordObject(control, "Edit Character Motion IK Target");
                control.SetPositionAndRotation(position, rotation);
                PrefabUtility.RecordPrefabInstancePropertyModifications(control);
                return;
            }

            CharacterMotionAuthoringUtility.GetChain(skeleton.GuideAnimator, selectedIk.Type,
                out Transform root, out Transform middle, out Transform tip);
            CharacterMotionIkChangeTracker.RunSuppressed(() =>
            {
                if (!poseEditActive)
                {
                    var undoObjects = new System.Collections.Generic.List<UnityEngine.Object> { control };
                    if (root != null) undoObjects.Add(root);
                    if (middle != null) undoObjects.Add(middle);
                    if (tip != null) undoObjects.Add(tip);
                    BeginPoseEdit(skeleton, "Edit Character Motion IK", undoObjects.ToArray());
                }
                control.SetPositionAndRotation(position, rotation);
                CharacterMotionAuthoringUtility.ApplyIk(keyframe, skeleton, selectedIk.Type);
            });
        }

        private void BeginPoseEdit(
            CharacterMotionSkeleton skeleton,
            string operation,
            params UnityEngine.Object[] undoObjects)
        {
            if (poseEditActive) return;
            Undo.IncrementCurrentGroup();
            poseEditUndoGroup = Undo.GetCurrentGroup();
            Undo.SetCurrentGroupName(operation);
            Undo.RecordObjects(undoObjects, operation);
            poseEditActive = true;
            pendingSkeleton = skeleton;
            pendingOperation = operation;
        }

        private void CompletePoseEditIfReleased()
        {
            if (!poseEditActive) return;
            Event current = Event.current;
            if (GUIUtility.hotControl != 0 || current != null && current.rawType == EventType.MouseDrag) return;
            FlushPendingPose();
        }

        internal void FlushPendingPose()
        {
            if (!poseEditActive) return;
            try
            {
                if (pendingSkeleton != null && keyframe != null)
                    CharacterMotionAuthoringUtility.Capture(pendingSkeleton, keyframe, pendingOperation);
                if (poseEditUndoGroup >= 0) Undo.CollapseUndoOperations(poseEditUndoGroup);
            }
            finally
            {
                poseEditActive = false;
                poseEditUndoGroup = -1;
                pendingSkeleton = null;
                pendingOperation = null;
            }
        }

        private static void DrawStoredIk(
            CharacterMotionKeyframe value,
            Vector3[] positions,
            Quaternion[] rotations,
            float armatureSize)
        {
            bool fullPose = CharacterMotionAuthoringUtility.HasFullPose(value);
            float size = IkSize(armatureSize);
            foreach (CharacterMotionConstraintType type in CharacterMotionAuthoringUtility.EffectorTypes)
            {
                bool enabled = (value.Constraints & type) != 0;
                Transform target = value.GetEffector(type);
                Transform pole = CharacterMotionAuthoringUtility.FindPole(value, type);
                bool invalid = enabled && (target == null ||
                    fullPose && (pole == null || IsUnreachable(type, target, positions)));
                Color color = invalid ? Invalid : enabled ? Enabled : Muted;
                if (target != null)
                {
                    IkBoxGeometry box = CalculateIkBox(type, target.position, target.rotation,
                        positions, rotations, armatureSize);
                    DrawBox(box, color);
                }
                if (fullPose && pole != null)
                {
                    int middle = type switch
                    {
                        CharacterMotionConstraintType.LeftHand => 12,
                        CharacterMotionConstraintType.RightHand => 18,
                        CharacterMotionConstraintType.LeftFoot => 23,
                        CharacterMotionConstraintType.RightFoot => 27,
                        _ => throw new ArgumentOutOfRangeException(nameof(type)),
                    };
                    DrawPole(pole.position, positions[middle], color, size * 0.39f);
                }
            }
        }

        private static bool IsJointEnabled(CharacterMotionConstraintType constraints, int joint)
        {
            if ((constraints & CharacterMotionConstraintType.FullBodyPose) != 0) return true;
            if (joint is >= 10 and <= 15 && (constraints & CharacterMotionConstraintType.LeftHand) != 0) return true;
            if (joint is >= 16 and <= 21 && (constraints & CharacterMotionConstraintType.RightHand) != 0) return true;
            if (joint is >= 22 and <= 25 && (constraints & CharacterMotionConstraintType.LeftFoot) != 0) return true;
            return joint is >= 26 and <= 29 && (constraints & CharacterMotionConstraintType.RightFoot) != 0;
        }

        private static bool IsUnreachable(CharacterMotionConstraintType type, Transform target, CharacterMotionSkeleton skeleton)
        {
            if (target == null) return true;
            CharacterMotionAuthoringUtility.GetChain(skeleton.GuideAnimator, type,
                out Transform root, out Transform middle, out Transform tip);
            if (root == null || middle == null || tip == null) return true;
            return OutsideReach(root.position, middle.position, tip.position, target.position);
        }

        private static bool IsUnreachable(CharacterMotionConstraintType type, Transform target, Vector3[] positions)
        {
            if (target == null) return true;
            (int root, int middle, int tip) chain = type switch
            {
                CharacterMotionConstraintType.LeftHand => (11, 12, 13),
                CharacterMotionConstraintType.RightHand => (17, 18, 19),
                CharacterMotionConstraintType.LeftFoot => (22, 23, 24),
                CharacterMotionConstraintType.RightFoot => (26, 27, 28),
                _ => throw new ArgumentOutOfRangeException(nameof(type)),
            };
            return OutsideReach(positions[chain.root], positions[chain.middle], positions[chain.tip], target.position);
        }

        private static bool OutsideReach(Vector3 root, Vector3 middle, Vector3 tip, Vector3 target)
        {
            float minimum = Mathf.Abs(Vector3.Distance(root, middle) - Vector3.Distance(middle, tip));
            float maximum = Vector3.Distance(root, middle) + Vector3.Distance(middle, tip);
            float distance = Vector3.Distance(root, target);
            return distance < minimum - 1e-4f || distance > maximum + 1e-4f;
        }

        internal static void DrawCapsule(Vector3 a, Vector3 b, Color color, float radius)
        {
            Vector3 direction = b - a;
            if (direction.sqrMagnitude < 1e-10f) return;
            Camera camera = SceneView.currentDrawingSceneView != null ? SceneView.currentDrawingSceneView.camera : Camera.current;
            Vector3 facing = camera != null ? camera.transform.forward : Vector3.forward;
            Vector3 side = Vector3.Cross(facing, direction).normalized;
            if (side.sqrMagnitude < 1e-8f) side = Vector3.Cross(Vector3.up, direction).normalized;
            Handles.color = color;
            Handles.DrawAAConvexPolygon(a + side * radius, b + side * radius, b - side * radius, a - side * radius);
            Handles.DrawSolidDisc(a, facing, radius);
            Handles.DrawSolidDisc(b, facing, radius);
        }

        internal static IkBoxGeometry CalculateIkBox(
            CharacterMotionConstraintType type,
            Vector3 center,
            Quaternion rotation,
            Vector3[] somaPositions,
            Quaternion[] somaRotations,
            float armatureSize)
        {
            (int first, int last) range = JointRange(type);
            int baseJoint = range.first;
            float minimumHalfExtent = IkSize(armatureSize) * 0.3f;
            var minimum = new Vector3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity);
            var maximum = new Vector3(float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity);
            Vector3 sourceCenter = somaPositions != null && baseJoint < somaPositions.Length &&
                                   CharacterMotionSpace.IsFinite(somaPositions[baseJoint])
                ? somaPositions[baseJoint]
                : center;
            Quaternion sourceRotation = somaRotations != null && baseJoint < somaRotations.Length &&
                                        CharacterMotionSpace.IsFinite(somaRotations[baseJoint])
                ? somaRotations[baseJoint]
                : rotation;
            Quaternion delta = rotation * Quaternion.Inverse(sourceRotation);
            Quaternion inverse = Quaternion.Inverse(rotation);
            bool hasPoint = false;
            if (somaPositions != null)
            {
                for (int i = range.first; i <= range.last && i < somaPositions.Length; i++)
                {
                    if (!CharacterMotionSpace.IsFinite(somaPositions[i])) continue;
                    Vector3 displayed = center + delta * (somaPositions[i] - sourceCenter);
                    Vector3 local = inverse * (displayed - center);
                    minimum = Vector3.Min(minimum, local);
                    maximum = Vector3.Max(maximum, local);
                    hasPoint = true;
                }
            }
            if (!hasPoint)
            {
                minimum = -Vector3.one * minimumHalfExtent;
                maximum = Vector3.one * minimumHalfExtent;
            }
            Vector3 localCenter = (minimum + maximum) * 0.5f;
            Vector3 halfExtents = (maximum - minimum) * 0.5f;
            halfExtents.x = Mathf.Max(halfExtents.x, minimumHalfExtent);
            halfExtents.y = Mathf.Max(halfExtents.y, minimumHalfExtent);
            halfExtents.z = Mathf.Max(halfExtents.z, minimumHalfExtent);
            float padding = Mathf.Max(BoneRadius(armatureSize) * 1.5f, armatureSize * 0.006f);
            return new IkBoxGeometry(center, center + rotation * localCenter, rotation,
                (halfExtents + Vector3.one * padding) * 2f);
        }

        internal static (int first, int last) JointRange(CharacterMotionConstraintType type) => type switch
        {
            CharacterMotionConstraintType.LeftHand => (13, 15),
            CharacterMotionConstraintType.RightHand => (19, 21),
            CharacterMotionConstraintType.LeftFoot => (24, 25),
            CharacterMotionConstraintType.RightFoot => (28, 29),
            _ => throw new ArgumentOutOfRangeException(nameof(type)),
        };

        internal static void DrawBox(IkBoxGeometry box, Color color)
        {
            using (new Handles.DrawingScope(color,
                       Matrix4x4.TRS(box.Center, box.Rotation, Vector3.one)))
                Handles.DrawWireCube(Vector3.zero, box.Size);
        }

        internal static float DistanceToBox(IkBoxGeometry box)
        {
            Vector3 half = box.Size * 0.5f;
            Span<Vector3> corners = stackalloc Vector3[8];
            for (int i = 0; i < corners.Length; i++)
            {
                var local = new Vector3(
                    (i & 1) == 0 ? -half.x : half.x,
                    (i & 2) == 0 ? -half.y : half.y,
                    (i & 4) == 0 ? -half.z : half.z);
                corners[i] = box.Center + box.Rotation * local;
            }

            float best = float.PositiveInfinity;
            for (int i = 0; i < corners.Length; i++)
            {
                for (int bit = 1; bit <= 4; bit <<= 1)
                {
                    int other = i ^ bit;
                    if (i > other) continue;
                    best = Mathf.Min(best, HandleUtility.DistanceToLine(corners[i], corners[other]));
                }
            }
            return best;
        }

        internal static void DrawPole(Vector3 position, Vector3 middleBone, Color color, float radius)
        {
            Camera camera = SceneView.currentDrawingSceneView != null
                ? SceneView.currentDrawingSceneView.camera
                : Camera.current;
            Vector3 normal = camera != null ? camera.transform.forward : Vector3.forward;
            Handles.color = color;
            Handles.DrawWireDisc(position, normal, radius);
            Handles.DrawAAPolyLine(2f, position, middleBone);
        }

        private static bool IsCurrent(IkPick pick, CharacterMotionConstraintType type, IkPart part)
            => pick.Type == type && pick.Part == part;

        private static string ControlLabel(CharacterMotionConstraintType type, IkPart part)
        {
            if (part == IkPart.Pole) return type + " Pole · Move";
            return Tools.current == Tool.Rotate ? type + " · Rotate (E)" : type + " · Move (W)";
        }

        private static Color Lighten(Color value) => Selected;
    }

    [InitializeOnLoad]
    internal static class CharacterMotionUnselectedSceneControls
    {
        private enum PickKind { Root, Bone, Ik }

        private readonly struct Pick
        {
            internal readonly CharacterMotionKeyframe Keyframe;
            internal readonly PickKind Kind;
            internal readonly int Joint;
            internal readonly CharacterMotionSceneHandles.IkPick Ik;

            internal Pick(CharacterMotionKeyframe keyframe, PickKind kind, int joint = -1,
                CharacterMotionSceneHandles.IkPick ik = default)
            {
                Keyframe = keyframe;
                Kind = kind;
                Joint = joint;
                Ik = ik;
            }
        }

        private static readonly List<CharacterMotionKeyframe> Keyframes = new List<CharacterMotionKeyframe>();
        private static readonly Dictionary<int, Pick> Picks = new Dictionary<int, Pick>();
        private static readonly Vector3[] Positions = new Vector3[30];
        private static readonly Quaternion[] Rotations = new Quaternion[30];
        private static bool cacheDirty = true;
        private static bool hasPendingSelection;
        private static EntityId pendingKeyframeId;
        private static int pendingJoint = -1;
        private static CharacterMotionSceneHandles.IkPick pendingIk;
        private static double pendingSelectionTime;

        static CharacterMotionUnselectedSceneControls()
        {
            SceneView.duringSceneGui += DuringSceneGui;
            EditorApplication.hierarchyChanged += InvalidateCache;
            Undo.undoRedoPerformed += InvalidateCache;
            EditorSceneManager.sceneOpened += OnSceneOpened;
            EditorSceneManager.sceneClosed += OnSceneClosed;
            EditorSceneManager.activeSceneChangedInEditMode += OnActiveSceneChanged;
        }

        internal static void ConsumeSelection(
            CharacterMotionKeyframe keyframe,
            out int joint,
            out CharacterMotionSceneHandles.IkPick ik)
        {
            joint = -1;
            ik = default;
            if (!hasPendingSelection || keyframe == null || keyframe.GetEntityId() != pendingKeyframeId) return;
            joint = pendingJoint;
            ik = pendingIk;
            hasPendingSelection = false;
            pendingKeyframeId = default;
            pendingJoint = -1;
            pendingIk = default;
            pendingSelectionTime = 0d;
        }

        private static void OnSceneOpened(Scene scene, OpenSceneMode mode) => InvalidateCache();

        private static void OnSceneClosed(Scene scene) => InvalidateCache();

        private static void OnActiveSceneChanged(Scene previous, Scene next) => InvalidateCache();

        private static void InvalidateCache()
        {
            cacheDirty = true;
            SceneView.RepaintAll();
        }

        private static void RefreshCache()
        {
            if (!cacheDirty) return;
            Keyframes.Clear();
            foreach (CharacterMotionKeyframe keyframe in Resources.FindObjectsOfTypeAll<CharacterMotionKeyframe>())
            {
                if (keyframe == null || EditorUtility.IsPersistent(keyframe) ||
                    !keyframe.gameObject.scene.IsValid() || !keyframe.gameObject.activeInHierarchy) continue;
                Keyframes.Add(keyframe);
            }
            Keyframes.Sort((a, b) => a.GetEntityId().GetHashCode().CompareTo(b.GetEntityId().GetHashCode()));
            cacheDirty = false;
        }

        private static void DuringSceneGui(SceneView sceneView)
        {
            Event current = Event.current;
            if (current == null) return;
            if (hasPendingSelection && EditorApplication.timeSinceStartup - pendingSelectionTime > 0.5d)
            {
                hasPendingSelection = false;
                pendingKeyframeId = default;
                pendingJoint = -1;
                pendingIk = default;
            }
            if (current.type == EventType.MouseDown)
            {
                HandleMouseDown(current);
                return;
            }
            if (current.type != EventType.Layout && current.type != EventType.Repaint) return;

            RefreshCache();
            if (current.type == EventType.Layout) Picks.Clear();
            for (int i = 0; i < Keyframes.Count; i++)
            {
                CharacterMotionKeyframe keyframe = Keyframes[i];
                if (keyframe == null || Selection.activeGameObject == keyframe.gameObject) continue;
                bool valid = ReconstructPose(keyframe);
                float armatureSize = CharacterMotionSceneHandles.MeasureArmature(Positions);
                if (current.type == EventType.Repaint)
                    CharacterMotionSceneHandles.DrawUnselectedKeyframe(
                        keyframe, Positions, Rotations, valid);
                else
                    RegisterControls(keyframe, armatureSize);
            }
        }

        private static bool ReconstructPose(CharacterMotionKeyframe keyframe)
            => CharacterMotionSceneHandles.TryReconstructPose(keyframe, Positions, Rotations);

        private static void RegisterControls(CharacterMotionKeyframe keyframe, float armatureSize)
        {
            Vector3 root = keyframe.transform.position;
            float rootSize = armatureSize * 0.025f;
            float rootDistance = Mathf.Min(
                HandleUtility.DistanceToCircle(root, rootSize),
                HandleUtility.DistanceToLine(root,
                    root + keyframe.transform.forward * armatureSize * 0.12f));
            Register(keyframe, new Pick(keyframe, PickKind.Root), rootDistance, 10f, 1f);

            bool fullPose = CharacterMotionAuthoringUtility.HasFullPose(keyframe);
            if (fullPose)
            {
                foreach ((int a, int b) in CharacterMotionSceneHandles.Segments)
                {
                    float distance = HandleUtility.DistanceToLine(Positions[a], Positions[b]);
                    Register(keyframe, new Pick(keyframe, PickKind.Bone, b), distance, 9f, 0.05f, 0.02f);
                }
            }

            foreach (CharacterMotionConstraintType type in CharacterMotionAuthoringUtility.EffectorTypes)
            {
                Transform target = keyframe.GetEffector(type);
                if (target != null)
                {
                    CharacterMotionSceneHandles.IkBoxGeometry box = CharacterMotionSceneHandles.CalculateIkBox(
                        type, target.position, target.rotation, Positions, Rotations, armatureSize);
                    float distance = CharacterMotionSceneHandles.DistanceToBox(box);
                    Register(keyframe, new Pick(keyframe, PickKind.Ik, ik:
                        new CharacterMotionSceneHandles.IkPick(type, CharacterMotionSceneHandles.IkPart.Target)),
                        distance, 10f, 0.75f);
                }

                if (!fullPose) continue;
                Transform pole = CharacterMotionAuthoringUtility.FindPole(keyframe, type);
                if (pole == null) continue;
                float poleDistance = HandleUtility.DistanceToCircle(
                    pole.position, CharacterMotionSceneHandles.IkSize(armatureSize) * 0.39f);
                Register(keyframe, new Pick(keyframe, PickKind.Ik, ik:
                    new CharacterMotionSceneHandles.IkPick(type, CharacterMotionSceneHandles.IkPart.Pole)),
                    poleDistance, 10f, 0.75f);
            }
        }

        private static void Register(
            CharacterMotionKeyframe keyframe,
            Pick pick,
            float distance,
            float maximumDistance,
            float priorityBias,
            float distanceScale = 1f)
        {
            if (float.IsNaN(distance) || float.IsInfinity(distance) || distance > maximumDistance) return;
            int hint = unchecked(keyframe.GetEntityId().GetHashCode() * 397 ^ (int)pick.Kind * 31 ^
                                 (pick.Joint + 1) * 17 ^ (int)pick.Ik.Type * 7 ^ (int)pick.Ik.Part);
            int controlId = GUIUtility.GetControlID(hint, FocusType.Passive);
            HandleUtility.AddControl(controlId, distance * distanceScale + priorityBias);
            Picks[controlId] = pick;
        }

        private static void HandleMouseDown(Event current)
        {
            if (current.button != 0 || current.alt || GUIUtility.hotControl != 0 ||
                !Picks.TryGetValue(HandleUtility.nearestControl, out Pick pick) || pick.Keyframe == null) return;
            hasPendingSelection = true;
            pendingKeyframeId = pick.Keyframe.GetEntityId();
            pendingJoint = pick.Kind == PickKind.Bone ? pick.Joint : -1;
            pendingIk = pick.Kind == PickKind.Ik ? pick.Ik : default;
            pendingSelectionTime = EditorApplication.timeSinceStartup;
            Selection.activeGameObject = pick.Keyframe.gameObject;
            current.Use();
            SceneView.RepaintAll();
        }
    }
}
