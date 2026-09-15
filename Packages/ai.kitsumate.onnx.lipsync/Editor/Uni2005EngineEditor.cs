#if UNITY_EDITOR
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.LipSync.Uni2005;
using KitsuMate.Onnx.Editor.Download;
using KitsuMate.Onnx.Download;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.LipSync.Editor
{
    [CustomEditor(typeof(Uni2005Engine))]
    public sealed class Uni2005EngineEditor : KitsuMate.Onnx.LipSync.Editor.LipSyncInferenceEditor
    {
        protected override void DrawEngineInspector()
        {
            DrawDefaultInspector();
            SerializedProperty modelSet = serializedObject.FindProperty("modelSet");
            if (modelSet.objectReferenceValue != null || !GUILayout.Button("Create Model Set and Download")) return;
            string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(target))?.Replace('\\', '/') ?? "Assets";
            var set = CreateInstance<Uni2005ModelSet>();
            AssetDatabase.CreateAsset(set, AssetDatabase.GenerateUniqueAssetPath($"{directory}/Uni2005ModelSet.asset"));
            modelSet.objectReferenceValue = set;
            serializedObject.ApplyModifiedProperties();
            OnnxSettings settings = OnnxSettings.Load();
            if (settings != null) settings.RegisterDefaultEngine((Uni2005Engine)target);
            AssetDatabase.SaveAssets();
            ModelDownloadWindow.Show(set);
        }
    }
}
#endif
