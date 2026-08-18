#if UNITY_EDITOR
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Embeddings;
using KitsuMate.Onnx.Editor.Download;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings.Editor
{
    [CustomEditor(typeof(TextEmbeddingModelSet))]
    public sealed class EmbeddingModelSetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update(); var set = (TextEmbeddingModelSet)target;
            ModelSetEditorUi.Header(set, "Text Embedding Model Set");
            EditorGUILayout.LabelField("ONNX Models", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_embeddingModelSource"), new GUIContent("Embedding Model"));
            EditorGUILayout.LabelField("Tokenizer", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_vocabulary"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_tokenizerModel"));
            EditorGUILayout.LabelField("Configuration", EditorStyles.boldLabel);
            foreach (string p in new[] { "_embeddingDimension", "_maxSequenceLength", "_useMeanPooling", "_normalizeEmbeddings" }) EditorGUILayout.PropertyField(serializedObject.FindProperty(p));
            serializedObject.ApplyModifiedProperties();
            if (GUILayout.Button("Download Models"))
                ShowDownload(set);
            ModelSetEditorUi.Validation(set);
        }

        internal static void ShowDownload(TextEmbeddingModelSet set)
        {
            ModelDownloadWindow.Show(new ModelDownloadRequest("KitsuMate/all-MiniLM-L6-v2-onnx",
                "d0c533e5999da1c893a0bba27d6336d423ba117d", "text-embedding"), result =>
            {
                result.RequireGraph("model", new[] { "input_ids", "attention_mask" });
                result.ConfigureModel(set.EmbeddingModel, "model");
                TextAsset vocabulary = result.ProjectPaths.ContainsKey("vocabulary") ? result.LoadAsset<TextAsset>("vocabulary") : null;
                TextAsset tokenizer = result.ProjectPaths.ContainsKey("tokenizer") ? result.LoadAsset<TextAsset>("tokenizer") : null;
                set.SetTokenizer(vocabulary, tokenizer);
                result.ApplyMetadata(set);
                AssetDatabase.SaveAssets();
                Selection.activeObject = set;
            });
        }

    }

    [CustomEditor(typeof(TextEmbeddingEngine))]
    public sealed class TextEmbeddingEngineEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            SerializedProperty modelSet = serializedObject.FindProperty("modelSet");
            if (modelSet.objectReferenceValue != null || !GUILayout.Button("Create Model Set and Download")) return;
            string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(target))?.Replace('\\', '/') ?? "Assets";
            var set = CreateInstance<TextEmbeddingModelSet>();
            AssetDatabase.CreateAsset(set, AssetDatabase.GenerateUniqueAssetPath($"{directory}/TextEmbeddingModelSet.asset"));
            modelSet.objectReferenceValue = set;
            serializedObject.ApplyModifiedProperties();
            OnnxSettings settings = OnnxSettings.Load();
            if (settings != null) settings.RegisterDefaultEngine((TextEmbeddingEngine)target);
            AssetDatabase.SaveAssets();
            EmbeddingModelSetEditor.ShowDownload(set);
        }
    }
}
#endif
