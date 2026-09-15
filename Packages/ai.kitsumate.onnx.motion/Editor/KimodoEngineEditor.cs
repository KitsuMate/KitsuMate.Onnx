#if UNITY_EDITOR
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Motion.Kimodo;
using KitsuMate.Onnx.Editor.Download;
using KitsuMate.Onnx.Download;
using KitsuMate.Onnx.Embeddings;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Editor
{
    [CustomEditor(typeof(KimodoEngine))]
    public sealed class KimodoEngineEditor : MotionInferenceEditor
    {
        protected override void DrawEngineInspector()
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
            Selection.activeObject = kimodo;
        }
    }
}
#endif
