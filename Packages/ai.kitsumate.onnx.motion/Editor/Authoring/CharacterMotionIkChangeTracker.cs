using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Editor
{
    [InitializeOnLoad]
    internal static class CharacterMotionIkChangeTracker
    {
        private sealed class PendingSync
        {
            internal CharacterMotionKeyframe Keyframe;
            internal CharacterMotionConstraintType Types;
            internal int UndoGroup;
        }

        private static readonly Dictionary<CharacterMotionKeyframe, PendingSync> Pending =
            new Dictionary<CharacterMotionKeyframe, PendingSync>();
        private static bool scheduled;
        private static bool suppressed;

        static CharacterMotionIkChangeTracker()
        {
            Undo.postprocessModifications += OnPostprocessModifications;
        }

        internal static void RunSuppressed(Action action)
        {
            bool previous = suppressed;
            suppressed = true;
            try { action(); }
            finally { suppressed = previous; }
        }

        internal static void SyncNow(CharacterMotionKeyframe keyframe, CharacterMotionConstraintType types)
        {
            if (keyframe == null || types == CharacterMotionConstraintType.None) return;
            if (!CharacterMotionAuthoringUtility.HasFullPose(keyframe)) return;
            CharacterMotionSkeleton skeleton = CharacterMotionAuthoringUtility.PrepareKeyframe(keyframe, false);
            bool applied = false;
            foreach (CharacterMotionConstraintType type in CharacterMotionAuthoringUtility.EffectorTypes)
            {
                if ((types & type) == 0 || (keyframe.Constraints & type) == 0) continue;
                CharacterMotionAuthoringUtility.GetChain(skeleton.GuideAnimator, type,
                    out Transform root, out Transform middle, out Transform tip);
                var undoObjects = new List<UnityEngine.Object> { keyframe };
                if (root != null) undoObjects.Add(root);
                if (middle != null) undoObjects.Add(middle);
                if (tip != null) undoObjects.Add(tip);
                Undo.RecordObjects(undoObjects.ToArray(), "Update Character Motion IK Pose");
                CharacterMotionAuthoringUtility.ApplyIk(keyframe, skeleton, type);
                applied = true;
            }
            if (!applied) return;
            CharacterMotionAuthoringUtility.Capture(skeleton, keyframe, "Update Character Motion IK Pose");
            PrefabUtility.RecordPrefabInstancePropertyModifications(keyframe);
        }

        private static UndoPropertyModification[] OnPostprocessModifications(UndoPropertyModification[] modifications)
        {
            if (suppressed || modifications == null) return modifications;
            foreach (UndoPropertyModification modification in modifications)
            {
                PropertyModification current = modification.currentValue;
                if (current?.target is not Transform transform || !IsTransformProperty(current.propertyPath)) continue;
                if (!TryResolveControl(transform, out CharacterMotionKeyframe keyframe,
                        out CharacterMotionConstraintType type)) continue;

                if (!Pending.TryGetValue(keyframe, out PendingSync sync))
                {
                    sync = new PendingSync
                    {
                        Keyframe = keyframe,
                        UndoGroup = Undo.GetCurrentGroup(),
                    };
                    Pending.Add(keyframe, sync);
                }
                sync.Types |= type;
            }

            if (Pending.Count > 0 && !scheduled)
            {
                scheduled = true;
                EditorApplication.delayCall += ProcessPending;
            }
            return modifications;
        }

        private static void ProcessPending()
        {
            scheduled = false;
            if (Pending.Count == 0) return;
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                Pending.Clear();
                return;
            }
            // Transform handles emit one modification per drag event. Wait until the
            // active control is released so IK is solved and serialized only once.
            if (GUIUtility.hotControl != 0)
            {
                scheduled = true;
                EditorApplication.delayCall += ProcessPending;
                return;
            }
            PendingSync[] values = new PendingSync[Pending.Count];
            Pending.Values.CopyTo(values, 0);
            Pending.Clear();

            RunSuppressed(() =>
            {
                foreach (PendingSync sync in values)
                {
                    if (sync.Keyframe == null) continue;
                    try
                    {
                        SyncNow(sync.Keyframe, sync.Types);
                        Undo.CollapseUndoOperations(sync.UndoGroup);
                    }
                    catch (Exception exception)
                    {
                        Debug.LogException(exception, sync.Keyframe);
                    }
                }
            });
            SceneView.RepaintAll();
        }

        private static bool TryResolveControl(
            Transform transform,
            out CharacterMotionKeyframe keyframe,
            out CharacterMotionConstraintType type)
        {
            keyframe = transform != null ? transform.GetComponentInParent<CharacterMotionKeyframe>() : null;
            if (keyframe != null && CharacterMotionAuthoringUtility.HasFullPose(keyframe))
            {
                foreach (CharacterMotionConstraintType candidate in CharacterMotionAuthoringUtility.EffectorTypes)
                {
                    if (keyframe.GetEffector(candidate) == transform ||
                        CharacterMotionAuthoringUtility.FindPole(keyframe, candidate) == transform)
                    {
                        type = candidate;
                        return true;
                    }
                }
            }
            type = CharacterMotionConstraintType.None;
            return false;
        }

        private static bool IsTransformProperty(string propertyPath)
            => !string.IsNullOrEmpty(propertyPath) &&
               (propertyPath.StartsWith("m_LocalPosition", StringComparison.Ordinal) ||
                propertyPath.StartsWith("m_LocalRotation", StringComparison.Ordinal));
    }
}
