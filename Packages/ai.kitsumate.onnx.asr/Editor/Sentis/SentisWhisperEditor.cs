using System;
using System.IO;
using KitsuMate.Onnx.Asr.Editor;
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Editor.Download;
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
            ModelDownloadWindow.Show(new ModelDownloadRequest(source.Repository, source.Revision, "fp32",
                ModelRoot(), "whisper"), result => Apply(set, result));
        }

        private static void Apply(SentisWhisperModelSet set, ModelDownloadResult result)
        {
            ModelAsset mel = result.LoadAsset<ModelAsset>("mel");
            ModelAsset encoder = result.LoadAsset<ModelAsset>("encoder");
            ModelAsset decoder = result.LoadAsset<ModelAsset>("decoder");
            ModelAsset decoderWithPast = result.LoadAsset<ModelAsset>("decoder-with-past");
            TextAsset tokenizer = result.LoadAsset<TextAsset>("tokenizer");
            string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(set))?.Replace('\\', '/') ?? "Assets";
            UnityAiInferenceModelAsset melSource = CreateSource(mel, directory);
            UnityAiInferenceModelAsset encoderSource = CreateSource(encoder, directory);
            UnityAiInferenceModelAsset decoderSource = CreateSource(decoder, directory);
            UnityAiInferenceModelAsset cachedDecoderSource = CreateSource(decoderWithPast, directory);
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

        private static string ModelRoot()
        {
            OnnxSettings settings = OnnxSettings.Load();
            return settings != null ? settings.ModelStorageRoot : "Assets/StreamingAssets/KitsuMateModels";
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
