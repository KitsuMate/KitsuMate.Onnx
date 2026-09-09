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
    [CustomEditor(typeof(WhisperModelSet))]
    public sealed class WhisperModelSetEditor : UnityEditor.Editor
    {
        private int source;

        public override void OnInspectorGUI()
        {
            serializedObject.Update(); var set = (WhisperModelSet)target;
            ModelSetEditorUi.Header(set, "Whisper Model Set");
            EditorGUILayout.LabelField("ONNX Models", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_melProcessorSource"), new GUIContent("Mel Processor"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_encoderSource"), new GUIContent("Encoder"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_decoderSource"), new GUIContent("Decoder"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_decoderWithPastSource"), new GUIContent("Decoder With Past"));
            EditorGUILayout.LabelField("Auxiliary Files", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_tokenizerJson"), new GUIContent("Tokenizer JSON"));
            serializedObject.ApplyModifiedProperties();

            EditorGUILayout.Space();
            string[] sources = System.Array.ConvertAll(WhisperModelDownloader.Sources, item => item.Name);
            source = EditorGUILayout.Popup("Model", Mathf.Clamp(source, 0, sources.Length - 1), sources);
            if (GUILayout.Button("Download Models"))
                WhisperModelDownloader.Show(set, WhisperModelDownloader.Sources[source]);

            ModelSetEditorUi.Validation(set);
        }
    }

    [CustomEditor(typeof(WhisperEngine))]
    public sealed class WhisperEngineEditor : UnityEditor.Editor
    {
        private int source;

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            SerializedProperty modelSet = serializedObject.FindProperty("modelSet");
            if (modelSet.objectReferenceValue != null) return;
            string[] names = System.Array.ConvertAll(WhisperModelDownloader.Sources, item => item.Name);
            source = EditorGUILayout.Popup("Model", Mathf.Clamp(source, 0, names.Length - 1), names);
            if (GUILayout.Button("Create Model Set and Download"))
            {
                var set = CreateModelSet<WhisperModelSet>((WhisperEngine)target, "WhisperModelSet");
                modelSet.objectReferenceValue = set;
                serializedObject.ApplyModifiedProperties();
                RegisterDefault((WhisperEngine)target);
                WhisperModelDownloader.Show(set, WhisperModelDownloader.Sources[source]);
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
