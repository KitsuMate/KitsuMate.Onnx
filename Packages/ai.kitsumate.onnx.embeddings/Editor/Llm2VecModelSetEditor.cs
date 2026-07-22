#if UNITY_EDITOR
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Embeddings;
using KitsuMate.Onnx.Editor.Download;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings.Editor
{
    [CustomEditor(typeof(Llm2VecModelSet))]
    public sealed class Llm2VecModelSetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update(); var set = (Llm2VecModelSet)target;
            ModelSetEditorUi.Header(set, "LLM2Vec Model Set");
            EditorGUILayout.LabelField("ONNX Models", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("encoderSource"), new GUIContent("Encoder"));
            EditorGUILayout.LabelField("Tokenizer and Contract", EditorStyles.boldLabel);
            foreach (string p in new[] { "tokenizerJson", "tokenizerConfig", "sequenceLength", "embeddingDimension" }) EditorGUILayout.PropertyField(serializedObject.FindProperty(p));
            serializedObject.ApplyModifiedProperties();
            if (GUILayout.Button("Download Models")) ShowDownload(set);
            ModelSetEditorUi.Validation(set);
        }

        internal static void ShowDownload(Llm2VecModelSet set)
        {
            ModelDownloadWindow.Show(new ModelDownloadRequest("KitsuMate/Llama-3-LLM2Vec-MNTP-Supervised-ONNX", "e00d62a8f3604bfab74b54a2a64f3b9fc1a13a74", string.Empty,
                ModelRoot(), "llm2vec"), result =>
            {
                result.ConfigureModel(set.Encoder, "encoder");
                TextAsset config = result.ProjectPaths.ContainsKey("tokenizer-config") ? result.LoadAsset<TextAsset>("tokenizer-config") : null;
                set.SetFiles(result.LoadAsset<TextAsset>("tokenizer"), config);
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

    [CustomEditor(typeof(Llm2VecEmbeddingEngine))]
    public sealed class Llm2VecEmbeddingEngineEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            SerializedProperty modelSet = serializedObject.FindProperty("modelSet");
            if (modelSet.objectReferenceValue != null || !GUILayout.Button("Create Model Set and Download")) return;
            string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(target))?.Replace('\\', '/') ?? "Assets";
            var set = CreateInstance<Llm2VecModelSet>();
            AssetDatabase.CreateAsset(set, AssetDatabase.GenerateUniqueAssetPath($"{directory}/Llm2VecModelSet.asset"));
            modelSet.objectReferenceValue = set;
            serializedObject.ApplyModifiedProperties();
            OnnxSettings settings = OnnxSettings.Load();
            if (settings != null) settings.RegisterDefaultEngine((Llm2VecEmbeddingEngine)target);
            AssetDatabase.SaveAssets();
            Llm2VecModelSetEditor.ShowDownload(set);
        }
    }
}
#endif
