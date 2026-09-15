#if UNITY_EDITOR
using System.IO;
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Editor.Download;
using KitsuMate.Onnx.Download;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.OmniVoice.Editor
{
    [CustomEditor(typeof(OmniVoiceEngine))]
    public sealed class OmniVoiceEngineEditor : KitsuMate.Onnx.Tts.Editor.TtsInferenceEditor
    {
        protected override bool SupportsVoiceDesign => true;

        protected override void DrawEngineInspector()
        {
            DrawDefaultInspector();
            SerializedProperty modelSet = serializedObject.FindProperty("modelSet");
            if (modelSet.objectReferenceValue != null || !GUILayout.Button("Create Model Set and Download")) return;
            string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(target))?.Replace('\\', '/') ?? "Assets";
            var set = CreateInstance<OmniVoiceModelSet>();
            AssetDatabase.CreateAsset(set, AssetDatabase.GenerateUniqueAssetPath($"{directory}/OmniVoiceModelSet.asset"));
            modelSet.objectReferenceValue = set;
            serializedObject.ApplyModifiedProperties();
            AssetDatabase.SaveAssets();
            Selection.activeObject = set;
        }
    }
}
#endif
