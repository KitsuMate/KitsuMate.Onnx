using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Embeddings;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Editor
{
    public abstract class MotionInferenceEditor : InferenceEngineEditor<CharacterMotionRequest, CharacterMotionResult>
    {
        [Serializable] private sealed class Inputs
        {
            public string Intent, Rig;
            public float Duration = 2;
            public string Output = "Assets/Generated/CharacterMotion/Inference.anim";
        }
        private readonly Inputs input = new();
        private GameObject capturedRig;
        private string capturedOutput;
        private AnimationClip output;
        protected override object TestInputs => input;
        protected override void DrawTestInputs()
        {
            input.Intent = InferenceTestPreferences.AssetField<CharacterMotionIntent>("Intent", input.Intent);
            input.Duration = EditorGUILayout.FloatField("Duration seconds", input.Duration);
            input.Rig = InferenceTestPreferences.AssetField<GameObject>("Humanoid rig prefab", input.Rig);
            input.Output = EditorGUILayout.TextField("Output asset", input.Output);
        }
        protected override Func<CancellationToken, Task<CharacterMotionRequest>> CaptureRequest()
        {
            var intent = InferenceTestPreferences.Asset<CharacterMotionIntent>(input.Intent);
            if (intent == null) throw new ArgumentException("Assign a Character Motion Intent.");
            capturedRig = InferenceTestPreferences.Asset<GameObject>(input.Rig);
            ValidateRig(capturedRig);
            capturedOutput = ValidateOutputPath(input.Output);
            var engine = (CharacterMotionEngine)Engine;
            if (!float.IsFinite(input.Duration) || input.Duration <= 0) throw new ArgumentException("Duration must be positive.");
            double frames = Math.Round(input.Duration * engine.ConstraintCapabilities.FramesPerSecond);
            if (frames < 2 || frames > int.MaxValue) throw new ArgumentException("Duration must produce at least two frames.");
            var settings = intent.Settings.CreateRequest((int)frames);
            var required = engine.RequiredEmbeddingModelIdentity;
            intent.TryGetEmbedding(required, out var baked);
            var embeddingEngine = intent.EmbeddingEngine;
            string prompt = intent.Prompt;
            if (baked == null && (string.IsNullOrWhiteSpace(prompt) || embeddingEngine == null || embeddingEngine.ModelSet == null ||
                embeddingEngine.ModelSet.Identity.ToString() != required.ToString()))
                throw new ArgumentException("The intent needs a valid baked embedding or a matching embedding engine and prompt.");
            return async token =>
            {
                var embedding = baked;
                if (embedding == null)
                {
                    var runtime = await embeddingEngine.CreateRuntimeAsync(token);
                    try
                    {
                        var result = await runtime.RunAsync(new EmbeddingRequest(prompt), token);
                        token.ThrowIfCancellationRequested();
                        embedding = new KimodoTextEmbedding(result.Embedding, required.ToString());
                    }
                    finally { await runtime.UnloadAsync(); runtime.Dispose(); }
                }
                return new CharacterMotionRequest { Segments = new[] { new CharacterMotionSegment(embedding, settings) } };
            };
        }
        public static string ValidateOutputPath(string path)
        {
            string full = Path.GetFullPath(path ?? "").Replace('\\', '/');
            string assets = Path.GetFullPath("Assets").Replace('\\', '/') + "/";
            if (!full.StartsWith(assets, StringComparison.OrdinalIgnoreCase) || !full.EndsWith(".anim", StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("Choose an .anim path inside Assets.");
            return "Assets/" + full.Substring(assets.Length);
        }
        private static void ValidateRig(GameObject prefab)
        {
            if (prefab == null || !PrefabUtility.IsPartOfPrefabAsset(prefab)) throw new ArgumentException("Assign a Humanoid rig prefab.");
            var animator = prefab.GetComponentInChildren<Animator>(true);
            if (animator == null || animator.avatar == null || !animator.avatar.isValid || !animator.avatar.isHuman)
                throw new ArgumentException("The rig requires an Animator with a valid Humanoid avatar.");
        }
        protected override void AcceptResult(CharacterMotionResult result) => output = ExportClip(capturedRig, result.Motion, capturedOutput);

        public static AnimationClip ExportClip(GameObject prefab, KimodoHumanoidMotion motion, string path)
        {
            ValidateRig(prefab);
            path = ValidateOutputPath(path);
            var scene = EditorSceneManager.NewPreviewScene();
            AnimationClip clip = null;
            try
            {
                var instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab, scene);
                instance.hideFlags = HideFlags.HideAndDontSave;
                instance.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
                var animator = instance.GetComponentInChildren<Animator>(true);
                var authoring = instance.GetComponent<CharacterMotion>() ?? instance.AddComponent<CharacterMotion>();
                authoring.ConfigureTarget(animator, instance.transform);
                // Force the export format on the temporary instance only.
                var serialized = new SerializedObject(authoring);
                serialized.FindProperty("clipFormat").enumValueIndex = (int)CharacterMotionClipFormat.Humanoid;
                serialized.ApplyModifiedPropertiesWithoutUndo();
                var skeleton = instance.GetComponentInChildren<CharacterMotionSkeleton>(true) ?? animator.gameObject.AddComponent<CharacterMotionSkeleton>();
                skeleton.Initialize(authoring, animator);
                clip = new AnimationClip { name = Path.GetFileNameWithoutExtension(path) };
                CharacterMotionClipBaker.PopulateClip(clip, authoring, motion);
                if (AnimationUtility.GetCurveBindings(clip).Length == 0) throw new InvalidOperationException("Generated animation contains no curves.");
                string folder = Path.GetDirectoryName(path).Replace('\\', '/');
                string current = "Assets";
                foreach (string part in folder.Substring("Assets".Length).Split(new[] { '/' }, StringSplitOptions.RemoveEmptyEntries))
                {
                    if (!AssetDatabase.IsValidFolder(current + "/" + part)) AssetDatabase.CreateFolder(current, part);
                    current += "/" + part;
                }
                AssetDatabase.CreateAsset(clip, AssetDatabase.GenerateUniqueAssetPath(path));
                AssetDatabase.SaveAssets();
                return clip;
            }
            finally
            {
                if (clip != null && !AssetDatabase.Contains(clip)) DestroyImmediate(clip);
                EditorSceneManager.ClosePreviewScene(scene);
            }
        }
        protected override void DrawTestResult()
        {
            EditorGUILayout.LabelField("Motion", $"{Result.Motion.FrameCount} frames at {Result.Motion.FramesPerSecond:F0} fps");
            EditorGUILayout.ObjectField("Animation", output, typeof(AnimationClip), false);
            if (GUILayout.Button("Show animation asset")) EditorGUIUtility.PingObject(output);
        }
    }
}
