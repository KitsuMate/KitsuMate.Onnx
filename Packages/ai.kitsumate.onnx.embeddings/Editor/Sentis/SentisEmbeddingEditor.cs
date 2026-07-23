#if UNITY_EDITOR
using System.IO;
using System;
using System.Linq;
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Editor.Download;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings.Sentis.Editor
{
    [CustomEditor(typeof(SentisEmbeddingModelSet))]
    public sealed class SentisEmbeddingModelSetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            var set = (SentisEmbeddingModelSet)target;
            if (GUILayout.Button("Download Models")) ShowDownload(set);
            ModelSetEditorUi.Validation(set);
        }

        internal static void ShowDownload(SentisEmbeddingModelSet set)
        {
            ModelDownloadWindow.ShowForSentis(new ModelDownloadRequest("KitsuMate/all-MiniLM-L6-v2-onnx",
                "9ec4eb6ad90ebff9e819a807468f37926836816f", "text-embedding"), result =>
            {
                ModelAsset imported = result.LoadAsset<ModelAsset>("model");
                Model model = ModelLoader.Load(imported);
                using (var worker = new Worker(model, BackendType.CPU)) { }
                string[] inputs = model.inputs.Select(input => input.name).ToArray();
                foreach (string input in new[] { "input_ids", "attention_mask" })
                    if (!inputs.Contains(input))
                        throw new InvalidOperationException(
                            $"Model '{imported.name}' is missing input '{input}'.");
                if (model.outputs.Count == 0)
                    throw new InvalidOperationException($"Model '{imported.name}' has no outputs.");
                string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(set))?.Replace('\\', '/') ?? "Assets";
                var source = CreateInstance<UnityAiInferenceModelAsset>();
                source.SetModelAsset(imported);
                AssetDatabase.CreateAsset(source, AssetDatabase.GenerateUniqueAssetPath($"{directory}/{imported.name}-Sentis.asset"));
                TextAsset vocabulary = result.ProjectPaths.ContainsKey("vocabulary") ? result.LoadAsset<TextAsset>("vocabulary") : null;
                TextAsset tokenizer = result.ProjectPaths.ContainsKey("tokenizer") ? result.LoadAsset<TextAsset>("tokenizer") : null;
                set.SetModels(source, vocabulary, tokenizer);
                result.ApplyMetadata(set);
                AssetDatabase.SaveAssets();
                Selection.activeObject = set;
            });
        }

    }

    [CustomEditor(typeof(SentisEmbeddingEngine))]
    public sealed class SentisEmbeddingEngineEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            SerializedProperty modelSet = serializedObject.FindProperty("modelSet");
            if (modelSet.objectReferenceValue != null || !GUILayout.Button("Create Model Set and Download")) return;
            string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(target))?.Replace('\\', '/') ?? "Assets";
            var set = CreateInstance<SentisEmbeddingModelSet>();
            AssetDatabase.CreateAsset(set, AssetDatabase.GenerateUniqueAssetPath($"{directory}/SentisEmbeddingModelSet.asset"));
            modelSet.objectReferenceValue = set;
            serializedObject.ApplyModifiedProperties();
            OnnxSettings settings = OnnxSettings.Load();
            if (settings != null) settings.RegisterDefaultEngine((SentisEmbeddingEngine)target);
            AssetDatabase.SaveAssets();
            SentisEmbeddingModelSetEditor.ShowDownload(set);
        }
    }
}
#endif
