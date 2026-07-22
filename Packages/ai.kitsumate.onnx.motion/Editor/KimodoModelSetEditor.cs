#if UNITY_EDITOR
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Motion.Kimodo;
using KitsuMate.Onnx.Editor.Download;
using KitsuMate.Onnx.Embeddings;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Editor
{
    [CustomEditor(typeof(KimodoModelSet))]
    public sealed class KimodoModelSetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            var modelSet = (KimodoModelSet)target;
            ModelSetEditorUi.Header(modelSet, "Kimodo Motion Model Set");
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("ONNX Models", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("motionModel"), new UnityEngine.GUIContent("Motion Model"));
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Embedding Contract", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("requiredEmbeddingModelSet"),
                new UnityEngine.GUIContent("Required Embedding Model"));
            serializedObject.ApplyModifiedProperties();
            if (GUILayout.Button("Download Models"))
                ShowDownload(modelSet);
            ModelSetEditorUi.Validation(modelSet);
        }

        internal static void ShowDownload(KimodoModelSet modelSet)
        {
            ModelDownloadWindow.Show(new ModelDownloadRequest("KitsuMate/Kimodo-SOMA-RP-v1.1-ONNX", "0c68c5599781a1a17cd61b1e0ed13f0a0029da90", string.Empty,
                ModelRoot(), "kimodo"), result =>
            {
                result.ConfigureModel(modelSet.MotionModel, "model");
                result.ApplyMetadata(modelSet);
                AssetDatabase.SaveAssets();
                Selection.activeObject = modelSet;
            });
        }

        internal static void ShowEmbeddingDownload(Llm2VecModelSet embedding, KimodoModelSet kimodo)
        {
            ModelDownloadWindow.Show(new ModelDownloadRequest("KitsuMate/Llama-3-LLM2Vec-MNTP-Supervised-ONNX", "e00d62a8f3604bfab74b54a2a64f3b9fc1a13a74", string.Empty,
                ModelRoot(), "llm2vec"), result =>
            {
                result.ConfigureModel(embedding.Encoder, "encoder");
                TextAsset config = result.ProjectPaths.ContainsKey("tokenizer-config") ? result.LoadAsset<TextAsset>("tokenizer-config") : null;
                embedding.SetFiles(result.LoadAsset<TextAsset>("tokenizer"), config);
                result.ApplyMetadata(embedding);
                AssetDatabase.SaveAssets();
                ShowDownload(kimodo);
            });
        }

        private static string ModelRoot()
        {
            OnnxSettings settings = OnnxSettings.Load();
            return settings != null ? settings.ModelStorageRoot : "Assets/StreamingAssets/KitsuMateModels";
        }
    }

    [CustomEditor(typeof(KimodoEngine))]
    public sealed class KimodoEngineEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            SerializedProperty modelSet = serializedObject.FindProperty("modelSet");
            if (modelSet.objectReferenceValue != null || !GUILayout.Button("Create Model Set and Download")) return;
            string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(target))?.Replace('\\', '/') ?? "Assets";
            var embedding = CreateInstance<Llm2VecModelSet>();
            var kimodo = CreateInstance<KimodoModelSet>();
            AssetDatabase.CreateAsset(embedding, AssetDatabase.GenerateUniqueAssetPath($"{directory}/Llm2VecModelSet.asset"));
            AssetDatabase.CreateAsset(kimodo, AssetDatabase.GenerateUniqueAssetPath($"{directory}/KimodoModelSet.asset"));
            kimodo.SetRequiredEmbedding(embedding);
            modelSet.objectReferenceValue = kimodo;
            serializedObject.ApplyModifiedProperties();
            OnnxSettings settings = OnnxSettings.Load();
            if (settings != null) settings.RegisterDefaultEngine((KimodoEngine)target);
            AssetDatabase.SaveAssets();
            KimodoModelSetEditor.ShowEmbeddingDownload(embedding, kimodo);
        }
    }
}
#endif
