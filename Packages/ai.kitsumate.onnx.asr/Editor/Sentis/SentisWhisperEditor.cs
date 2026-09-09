using System;
using System.IO;
using System.Linq;
using KitsuMate.Onnx.Asr.Editor;
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Editor.Download;
using KitsuMate.Onnx.Download;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Asr.Sentis.Editor
{
    [CustomEditor(typeof(SentisWhisperModelSet))]
    public sealed class SentisWhisperModelSetEditor : UnityEditor.Editor
    {
        private int source;

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            var set = (SentisWhisperModelSet)target;
            string[] names = Array.ConvertAll(WhisperModelDownloader.Sources, item => item.Name);
            source = EditorGUILayout.Popup("Model", Mathf.Clamp(source, 0, names.Length - 1), names);
            if (GUILayout.Button("Download Models"))
                ShowDownload(set, WhisperModelDownloader.Sources[source]);
            ModelSetEditorUi.Validation(set);
        }

        internal static void ShowDownload(SentisWhisperModelSet set, WhisperModelDownloader.Source source)
        {
            ModelDownloadWindow.ShowForSentis(
                new ModelDownloadRequest(source.Repository, source.Revision, "whisper"),
                result => Apply(set, result));
        }

        private static void Apply(SentisWhisperModelSet set, ModelDownloadResult result)
        {
            ModelAsset mel = result.LoadAsset<ModelAsset>("mel");
            ModelAsset encoder = result.LoadAsset<ModelAsset>("encoder");
            ModelAsset decoder = result.LoadAsset<ModelAsset>("decoder");
            ModelAsset decoderWithPast = result.ProjectPaths.ContainsKey("decoder-with-past")
                ? result.LoadAsset<ModelAsset>("decoder-with-past")
                : null;
            TextAsset tokenizer = result.LoadAsset<TextAsset>("tokenizer");
            Validate(mel, new[] { "audio" }, new[] { "log_mel" });
            ValidateAnyInput(encoder, new[] { "mel", "input_features" }, "last_hidden_state");
            Model decoderModel = Validate(decoder,
                new[] { "input_ids", "encoder_hidden_states" }, new[] { "logits" });
            Model cachedModel = decoderWithPast != null
                ? Validate(decoderWithPast, new[] { "input_ids" }, new[] { "logits" })
                : null;
            ValidateCache(decoderModel, cachedModel);
            string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(set))?.Replace('\\', '/') ?? "Assets";
            UnityAiInferenceModelAsset melSource = CreateSource(mel, directory);
            UnityAiInferenceModelAsset encoderSource = CreateSource(encoder, directory);
            UnityAiInferenceModelAsset decoderSource = CreateSource(decoder, directory);
            UnityAiInferenceModelAsset cachedDecoderSource =
                decoderWithPast != null ? CreateSource(decoderWithPast, directory) : null;
            set.SetModels(melSource, encoderSource, decoderSource, cachedDecoderSource, tokenizer);
            result.ApplyMetadata(set);
            AssetDatabase.SaveAssets();
            Selection.activeObject = set;
        }

        private static UnityAiInferenceModelAsset CreateSource(ModelAsset model, string directory)
        {
            var source = CreateInstance<UnityAiInferenceModelAsset>();
            source.SetModelAsset(model);
            string path = AssetDatabase.GenerateUniqueAssetPath($"{directory}/{model.name}-Sentis.asset");
            AssetDatabase.CreateAsset(source, path);
            return source;
        }

        private static Model Validate(ModelAsset asset, string[] requiredInputs, string[] requiredOutputs)
        {
            Model model = ModelLoader.Load(asset);
            using var worker = new Worker(model, BackendType.CPU);
            string[] inputs = model.inputs.Select(input => input.name).ToArray();
            string[] outputs = model.outputs.Select(output => output.name).ToArray();
            foreach (string input in requiredInputs)
                if (!inputs.Contains(input))
                    throw new InvalidOperationException($"Model '{asset.name}' is missing input '{input}'.");
            foreach (string output in requiredOutputs)
                if (!outputs.Contains(output))
                    throw new InvalidOperationException($"Model '{asset.name}' is missing output '{output}'.");
            return model;
        }

        private static void ValidateAnyInput(ModelAsset asset, string[] inputNames, string outputName)
        {
            Model model = ModelLoader.Load(asset);
            using var worker = new Worker(model, BackendType.CPU);
            if (!model.inputs.Any(input => inputNames.Contains(input.name)))
                throw new InvalidOperationException(
                    $"Model '{asset.name}' is missing input '{string.Join("' or '", inputNames)}'.");
            if (!model.outputs.Any(output => output.name == outputName))
                throw new InvalidOperationException($"Model '{asset.name}' is missing output '{outputName}'.");
        }

        private static void ValidateCache(Model decoder, Model cachedDecoder)
        {
            if (cachedDecoder == null)
            {
                if (!decoder.inputs.Any(input => input.name == "use_cache_branch"))
                    throw new InvalidOperationException(
                        "A Whisper decoder without decoder-with-past must provide 'use_cache_branch'.");
                return;
            }

            int initialCacheCount = decoder.outputs.Count(output =>
                output.name.StartsWith("present.", StringComparison.Ordinal));
            int cacheInputCount = cachedDecoder.inputs.Count(input =>
                input.name.StartsWith("past_key_values.", StringComparison.Ordinal));
            int cacheOutputCount = cachedDecoder.outputs.Count(output =>
                output.name.StartsWith("present.", StringComparison.Ordinal));
            if (initialCacheCount == 0 || initialCacheCount != cacheInputCount ||
                cacheInputCount != cacheOutputCount)
                throw new InvalidOperationException(
                    "Whisper initial and cached decoder cache contracts do not match.");
        }
    }

    [CustomEditor(typeof(SentisWhisperEngine))]
    public sealed class SentisWhisperEngineEditor : UnityEditor.Editor
    {
        private int source;

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            SerializedProperty modelSet = serializedObject.FindProperty("modelSet");
            if (modelSet.objectReferenceValue != null) return;
            string[] names = Array.ConvertAll(WhisperModelDownloader.Sources, item => item.Name);
            source = EditorGUILayout.Popup("Model", Mathf.Clamp(source, 0, names.Length - 1), names);
            if (!GUILayout.Button("Create Model Set and Download")) return;
            string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(target))?.Replace('\\', '/') ?? "Assets";
            var set = CreateInstance<SentisWhisperModelSet>();
            AssetDatabase.CreateAsset(set, AssetDatabase.GenerateUniqueAssetPath($"{directory}/SentisWhisperModelSet.asset"));
            modelSet.objectReferenceValue = set;
            serializedObject.ApplyModifiedProperties();
            OnnxSettings settings = OnnxSettings.Load();
            if (settings != null) settings.RegisterDefaultEngine((SentisWhisperEngine)target);
            AssetDatabase.SaveAssets();
            SentisWhisperModelSetEditor.ShowDownload(set, WhisperModelDownloader.Sources[source]);
        }
    }
}
