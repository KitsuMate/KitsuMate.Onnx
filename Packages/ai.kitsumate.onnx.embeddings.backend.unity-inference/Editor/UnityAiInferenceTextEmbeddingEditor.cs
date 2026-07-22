#if UNITY_EDITOR
using System.IO;
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Editor.Download;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings.UnityAiInference.Editor
{
    [CustomEditor(typeof(UnityAiInferenceTextEmbeddingModelSet))]
    public sealed class UnityAiInferenceTextEmbeddingModelSetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            var set = (UnityAiInferenceTextEmbeddingModelSet)target;
            if (GUILayout.Button("Download Models")) ShowDownload(set);
            ModelSetEditorUi.Validation(set);
        }

        internal static void ShowDownload(UnityAiInferenceTextEmbeddingModelSet set)
        {
            ModelDownloadWindow.Show(new ModelDownloadRequest("KitsuMate/all-MiniLM-L6-v2-onnx", "caa577297ce0c15e94a2116d2f9228c640d1224c", "unity-fp32",
                ModelRoot(), "text-embedding"), result =>
            {
                ModelAsset imported = result.LoadAsset<ModelAsset>("model");
                string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(set))?.Replace('\\', '/') ?? "Assets";
                var source = CreateInstance<UnityAiInferenceModelAsset>();
                source.SetModelAsset(imported);
                AssetDatabase.CreateAsset(source, AssetDatabase.GenerateUniqueAssetPath($"{directory}/{imported.name}-UnityInference.asset"));
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

    [CustomEditor(typeof(UnityAiInferenceTextEmbeddingEngine))]
    public sealed class UnityAiInferenceTextEmbeddingEngineEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            SerializedProperty modelSet = serializedObject.FindProperty("modelSet");
            if (modelSet.objectReferenceValue != null || !GUILayout.Button("Create Model Set and Download")) return;
            string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(target))?.Replace('\\', '/') ?? "Assets";
            var set = CreateInstance<UnityAiInferenceTextEmbeddingModelSet>();
            AssetDatabase.CreateAsset(set, AssetDatabase.GenerateUniqueAssetPath($"{directory}/UnityAiInferenceTextEmbeddingModelSet.asset"));
            modelSet.objectReferenceValue = set;
            serializedObject.ApplyModifiedProperties();
            OnnxSettings settings = OnnxSettings.Load();
            if (settings != null) settings.RegisterDefaultEngine((UnityAiInferenceTextEmbeddingEngine)target);
            AssetDatabase.SaveAssets();
            UnityAiInferenceTextEmbeddingModelSetEditor.ShowDownload(set);
        }
    }
}
#endif
