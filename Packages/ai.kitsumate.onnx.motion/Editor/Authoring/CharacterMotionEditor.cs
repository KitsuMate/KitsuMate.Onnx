using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Editor
{
    [CustomEditor(typeof(CharacterMotion))]
    internal sealed class CharacterMotionEditor : UnityEditor.Editor
    {
        private float previewTime;
        private bool ownsPreview;
        private SerializedProperty parts;
        private SerializedProperty previousMotion;
        private SerializedProperty entryOverlapFrames;
        private SerializedProperty internalOverlapFrames;
        private SerializedProperty targetAnimator;
        private SerializedProperty actionOrigin;
        private SerializedProperty overrideGenerationSettings;
        private SerializedProperty generationSettings;
        private SerializedProperty clipFormat;
        private SerializedProperty bakedClip;
        private SerializedProperty motionEngine;

        private void OnEnable()
        {
            parts = serializedObject.FindProperty("parts");
            previousMotion = serializedObject.FindProperty("previousMotion");
            entryOverlapFrames = serializedObject.FindProperty("entryOverlapFrames");
            internalOverlapFrames = serializedObject.FindProperty("internalOverlapFrames");
            targetAnimator = serializedObject.FindProperty("targetAnimator");
            actionOrigin = serializedObject.FindProperty("actionOrigin");
            overrideGenerationSettings = serializedObject.FindProperty("overrideGenerationSettings");
            generationSettings = serializedObject.FindProperty("generationSettings");
            clipFormat = serializedObject.FindProperty("clipFormat");
            bakedClip = serializedObject.FindProperty("bakedClip");
            motionEngine = serializedObject.FindProperty("motionEngine");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EditorGUILayout.LabelField("Motion", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(targetAnimator);
            EditorGUILayout.PropertyField(actionOrigin);
            EditorGUILayout.PropertyField(previousMotion);
            if (previousMotion.objectReferenceValue != null)
                EditorGUILayout.PropertyField(entryOverlapFrames, new GUIContent("Entry Overlap Frames"));

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Intent Timeline", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(parts, true);
            if (ExpandedRunCount(parts) > 1)
                EditorGUILayout.PropertyField(internalOverlapFrames, new GUIContent("Internal Overlap Frames"));

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Generation", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(overrideGenerationSettings, new GUIContent("Override Generation Settings"));
            if (overrideGenerationSettings.boolValue)
                EditorGUILayout.PropertyField(generationSettings, true);
            EditorGUILayout.PropertyField(clipFormat);
            EditorGUILayout.PropertyField(motionEngine);
            using (new EditorGUI.DisabledScope(true)) EditorGUILayout.PropertyField(bakedClip);
            serializedObject.ApplyModifiedProperties();
            var motion = (CharacterMotion)target;

            try
            {
                CharacterMotionTimeline timeline = motion.Timeline;
                EditorGUILayout.HelpBox(
                    $"Inference runs: {timeline.RunCount}\nClip frames: {timeline.FrameCount} " +
                    $"({timeline.FrameCount / 30f:F2} s at 30 FPS)" +
                    (motion.PreviousMotion != null
                        ? $"\nEffective chained frames: {timeline.EffectiveChainedFrameCount}"
                        : string.Empty), MessageType.Info);
            }
            catch (Exception exception)
            {
                EditorGUILayout.HelpBox(exception.Message, MessageType.Error);
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Authoring", EditorStyles.boldLabel);
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Create/Rebuild Skeleton")) Run(() =>
                {
                    StopPreview();
                    ownsPreview = false;
                    CharacterMotionSkeletonUtility.Rebuild(motion);
                });
                if (GUILayout.Button("Remove Skeleton")) CharacterMotionSkeletonUtility.Remove(motion);
            }
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Add Root")) AddKeyframe(motion, CharacterMotionConstraintType.RootPosition | CharacterMotionConstraintType.RootHeading);
                if (GUILayout.Button("Add Pose")) AddKeyframe(motion, CharacterMotionConstraintType.FullBodyPose);
                if (GUILayout.Button("Add Effectors")) AddKeyframe(motion, CharacterMotionConstraintType.LeftHand | CharacterMotionConstraintType.RightHand);
            }

            CharacterMotionValidationResult validation = motion.ValidateMotion();
            foreach (CharacterMotionDiagnostic diagnostic in validation.Diagnostics)
                EditorGUILayout.HelpBox(diagnostic.Message,
                    diagnostic.Severity == CharacterMotionDiagnosticSeverity.Error ? MessageType.Error : MessageType.Warning);

            var bakeDiagnostics = new List<string>();
            bool animationReady = validation.IsValid && ValidateAnimationConfiguration(motion, bakeDiagnostics);
            foreach (string diagnostic in bakeDiagnostics.Distinct())
                EditorGUILayout.HelpBox(diagnostic, MessageType.Error);
            CharacterMotionIntent missingEmbedding = FirstMissingEmbedding(motion);
            if (missingEmbedding != null && GUILayout.Button("Select Intent to Bake Embedding"))
                Selection.activeObject = missingEmbedding;

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Bake", EditorStyles.boldLabel);
            using (new EditorGUI.DisabledScope(!animationReady))
                if (GUILayout.Button("Bake Animation")) _ = RunAsync(() => CharacterMotionClipBaker.BakeAnimationAsync(motion));

            if (motion.BakedClip != null)
            {
                EditorGUI.BeginChangeCheck();
                previewTime = EditorGUILayout.Slider("Preview Time", previewTime, 0f, motion.BakedClip.length);
                if (EditorGUI.EndChangeCheck()) Run(() =>
                {
                    Preview(motion, previewTime);
                    ownsPreview = true;
                });
                using (new EditorGUILayout.HorizontalScope())
                {
                    if (GUILayout.Button("Stop Preview"))
                    {
                        StopPreview();
                        ownsPreview = false;
                    }
                    if (GUILayout.Button("Clear Clip"))
                    {
                        Undo.RecordObject(motion, "Clear Baked Character Motion");
                        motion.SetBakedClip(null, null, null);
                        EditorUtility.SetDirty(motion);
                    }
                }
            }
        }

        private static bool ValidateAnimationConfiguration(CharacterMotion motion, List<string> diagnostics)
        {
            if (motion.MotionEngine == null)
            {
                diagnostics.Add("Assign a Character Motion engine.");
                return false;
            }
            if (motion.MotionEngine.ModelSet == null)
            {
                diagnostics.Add($"Motion engine '{motion.MotionEngine.name}' has no model set assigned.");
                return false;
            }
            if (!motion.MotionEngine.ModelSet.IsComplete)
            {
                diagnostics.Add($"Motion model set '{motion.MotionEngine.ModelSet.name}' is incomplete.");
                return false;
            }
            if (motion.MotionEngine.Backend == null)
            {
                diagnostics.Add($"Motion engine '{motion.MotionEngine.name}' has no backend assigned.");
                return false;
            }
            ModelValidationResult result = motion.MotionEngine.ModelSet.Validate(new ModelValidationContext(motion.MotionEngine.Backend));
            foreach (ModelDiagnostic value in result.Diagnostics.Where(value => value.Severity == ModelDiagnosticSeverity.Error))
                diagnostics.Add(value.Message);
            if (!result.IsValid) return false;
            foreach (CharacterMotionIntent intent in motion.Parts.Where(value => value != null)
                         .Select(value => value.Intent).Where(value => value != null).Distinct())
            {
                if (intent.TryGetEmbedding(motion.RequiredEmbeddingModelIdentity, out _)) continue;
                diagnostics.Add(
                    $"Intent '{intent.name}' has no current embedding compatible with '{motion.RequiredEmbeddingModelIdentity}'. " +
                    "Select the intent and bake its embedding first.");
            }
            return diagnostics.Count == 0;
        }

        private static CharacterMotionIntent FirstMissingEmbedding(CharacterMotion motion)
        {
            if (motion.MotionEngine == null) return null;
            return motion.Parts.Where(value => value != null).Select(value => value.Intent)
                .FirstOrDefault(intent => intent != null &&
                    !intent.TryGetEmbedding(motion.RequiredEmbeddingModelIdentity, out _));
        }

        private static int ExpandedRunCount(SerializedProperty values)
        {
            int result = 0;
            for (int i = 0; i < values.arraySize; i++)
            {
                SerializedProperty part = values.GetArrayElementAtIndex(i);
                SerializedProperty repetitions = part.FindPropertyRelative("repetitions");
                result += Mathf.Max(1, repetitions != null ? repetitions.intValue : 1);
            }
            return result;
        }

        private static void AddKeyframe(CharacterMotion motion, CharacterMotionConstraintType type)
        {
            Transform container = motion.transform.Find("Keyframes");
            if (container == null)
            {
                var group = new GameObject("Keyframes");
                Undo.RegisterCreatedObjectUndo(group, "Create Character Motion Keyframes");
                group.transform.SetParent(motion.transform, false);
                container = group.transform;
            }
            int[] used = motion.GetKeyframes().Select(value => value.Frame).ToArray();
            int frameCount = Mathf.Max(1, motion.Timeline.FrameCount);
            int frame = Enumerable.Range(0, frameCount).FirstOrDefault(value => !used.Contains(value));
            var gameObject = new GameObject($"Frame_{frame:00}");
            Undo.RegisterCreatedObjectUndo(gameObject, "Add Character Motion Keyframe");
            gameObject.transform.SetParent(container, false);
            gameObject.transform.SetPositionAndRotation(motion.ActionOrigin.position, motion.ActionOrigin.rotation);
            CharacterMotionKeyframe keyframe = Undo.AddComponent<CharacterMotionKeyframe>(gameObject);
            keyframe.Configure(frame, type);
            Selection.activeGameObject = gameObject;
        }

        internal static void Preview(CharacterMotion motion, float time)
        {
            CharacterMotionSkeleton skeleton = motion.GetComponentInChildren<CharacterMotionSkeleton>(true);
            if (skeleton == null || skeleton.GuideAnimator == null)
                throw new InvalidOperationException("Create the Character Motion skeleton before previewing a baked clip.");

            // End the active preview before sampling another clip.
            StopPreview();
            skeleton.Preview(motion.BakedClip, time);
            SceneView.RepaintAll();
        }

        internal static void StopPreview()
        {
            if (AnimationMode.InAnimationMode()) AnimationMode.StopAnimationMode();
            foreach (CharacterMotionSkeleton skeleton in UnityEngine.Object.FindObjectsByType<CharacterMotionSkeleton>(
                         FindObjectsInactive.Include, FindObjectsSortMode.None))
                skeleton.StopPreview();
            SceneView.RepaintAll();
        }

        private void OnDisable()
        {
            if (!ownsPreview) return;
            StopPreview();
            ownsPreview = false;
        }

        private static void Run(Action action)
        {
            try { action(); }
            catch (Exception exception) { Debug.LogException(exception); EditorUtility.DisplayDialog("Character Motion", exception.Message, "OK"); }
        }

        private static async Awaitable RunAsync(Func<Awaitable> action)
        {
            try { await action(); }
            catch (Exception exception) { Debug.LogException(exception); EditorUtility.DisplayDialog("Character Motion", exception.Message, "OK"); }
        }
    }
}
