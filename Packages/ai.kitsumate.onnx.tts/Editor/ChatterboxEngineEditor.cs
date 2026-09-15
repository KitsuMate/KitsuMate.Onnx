#if UNITY_EDITOR
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Editor.Download;
using KitsuMate.Onnx.Download;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Chatterbox.Editor
{
    [CustomEditor(typeof(ChatterboxEngine))]
    public sealed class ChatterboxEngineEditor : KitsuMate.Onnx.Tts.Editor.TtsInferenceEditor
    {
        protected override void DrawEngineInspector()
        {
            DrawDefaultInspector();
            SerializedProperty modelSet = serializedObject.FindProperty("modelSet");
            if (modelSet.objectReferenceValue != null || !GUILayout.Button("Create Model Set and Download")) return;
            string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(target))?.Replace('\\', '/') ?? "Assets";
            var set = CreateInstance<ChatterboxModelSet>();
            AssetDatabase.CreateAsset(set, AssetDatabase.GenerateUniqueAssetPath($"{directory}/ChatterboxModelSet.asset"));
            modelSet.objectReferenceValue = set;
            serializedObject.ApplyModifiedProperties();
            OnnxSettings settings = OnnxSettings.Load();
            if (settings != null) settings.RegisterDefaultEngine((ChatterboxEngine)target);
            AssetDatabase.SaveAssets();
            ModelDownloadWindow.Show(set);
        }
    }
}
#endif
