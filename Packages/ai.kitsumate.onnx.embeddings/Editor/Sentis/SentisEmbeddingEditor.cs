#if UNITY_EDITOR
using System.IO;
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
            ModelDownloadWindow.Show(new ModelDownloadRequest("KitsuMate/all-MiniLM-L6-v2-onnx", "caa577297ce0c15e94a2116d2f9228c640d1224c", "unity-fp32",
                ModelRoot(), "text-embedding"), result =>
            {
                ModelAsset imported = result.LoadAsset<ModelAsset>("model");
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

        private static string ModelRoot()
        {
            OnnxSettings settings = OnnxSettings.Load();
            return settings != null ? settings.ModelStorageRoot : "Assets/StreamingAssets/KitsuMateModels";
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
