#if UNITY_EDITOR
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Embeddings;
using KitsuMate.Onnx.Editor.Download;
using KitsuMate.Onnx.Download;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings.Editor
{
    [CustomEditor(typeof(Llm2VecEmbeddingEngine))]
    public sealed class Llm2VecEmbeddingEngineEditor : KitsuMate.Onnx.Embeddings.Editor.EmbeddingInferenceEditor
    {
        protected override void DrawEngineInspector()
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
            ModelDownloadWindow.Show(set);
        }
    }
}
#endif
