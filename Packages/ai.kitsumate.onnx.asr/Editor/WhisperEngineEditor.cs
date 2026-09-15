#if UNITY_EDITOR
using KitsuMate.Onnx.Asr.Whisper;
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Editor.Download;
using KitsuMate.Onnx.Download;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Asr.Editor
{
    [CustomEditor(typeof(WhisperEngine))]
    public sealed class WhisperEngineEditor : KitsuMate.Onnx.Asr.Editor.AsrInferenceEditor
    {

        protected override void DrawEngineInspector()
        {
            DrawDefaultInspector();
            SerializedProperty modelSet = serializedObject.FindProperty("modelSet");
            if (modelSet.objectReferenceValue != null) return;
            if (GUILayout.Button("Create Model Set and Download"))
            {
                var set = CreateModelSet<WhisperModelSet>((WhisperEngine)target, "WhisperModelSet");
                modelSet.objectReferenceValue = set;
                serializedObject.ApplyModifiedProperties();
                RegisterDefault((WhisperEngine)target);
                ModelDownloadWindow.Show(set);
            }
        }

        private static T CreateModelSet<T>(UnityEngine.Object engine, string name) where T : ScriptableObject
        {
            string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(engine))?.Replace('\\', '/') ?? "Assets";
            string path = AssetDatabase.GenerateUniqueAssetPath($"{directory}/{name}.asset");
            T set = CreateInstance<T>();
            AssetDatabase.CreateAsset(set, path);
            AssetDatabase.SaveAssets();
            return set;
        }

        private static void RegisterDefault(WhisperEngine engine)
        {
            OnnxSettings settings = OnnxSettings.Load();
            if (settings != null && settings.RegisterDefaultEngine(engine)) AssetDatabase.SaveAssets();
        }
    }
}
#endif
