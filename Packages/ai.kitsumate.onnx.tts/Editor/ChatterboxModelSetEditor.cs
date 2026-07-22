#if UNITY_EDITOR
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Editor.Download;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Chatterbox.Editor
{
    [CustomEditor(typeof(ChatterboxModelSet))]
    public sealed class ChatterboxModelSetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update(); var set = (ChatterboxModelSet)target;
            ModelSetEditorUi.Header(set, "Chatterbox Model Set");
            EditorGUILayout.LabelField("ONNX Models", EditorStyles.boldLabel);
            foreach ((string property, string label) in new[] { ("_speechEncoderSource", "Speech Encoder"), ("_embedTokensSource", "Embed Tokens"), ("_languageModelSource", "Language Model"), ("_conditionalDecoderSource", "Conditional Decoder") })
                EditorGUILayout.PropertyField(serializedObject.FindProperty(property), new GUIContent(label));
            EditorGUILayout.LabelField("Auxiliary Files", EditorStyles.boldLabel);
            foreach (string p in new[] { "_tokenizer", "_cangjieMapping", "_defaultVoice" }) EditorGUILayout.PropertyField(serializedObject.FindProperty(p));
            serializedObject.ApplyModifiedProperties();
            if (GUILayout.Button("Download Models"))
                ShowDownload(set);
            ModelSetEditorUi.Validation(set);
        }

        internal static void ShowDownload(ChatterboxModelSet set)
        {
            ModelDownloadWindow.Show(new ModelDownloadRequest("KitsuMate/chatterbox-onnx", "f1c7a1ca9ad781ecf6fa60a6b2bcda8f4e316f70", string.Empty,
                ModelRoot(), "chatterbox"), result =>
            {
                result.ConfigureModel(set.SpeechEncoder, "speech-encoder");
                result.ConfigureModel(set.EmbedTokens, "embed-tokens");
                result.ConfigureModel(set.LanguageModel, "language-model");
                result.ConfigureModel(set.ConditionalDecoder, "conditional-decoder");
                TextAsset mapping = result.ProjectPaths.ContainsKey("cangjie") ? result.LoadAsset<TextAsset>("cangjie") : null;
                set.SetFiles(result.LoadAsset<TextAsset>("tokenizer"), mapping, result.LoadAsset<AudioClip>("voice"));
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

    [CustomEditor(typeof(ChatterboxEngine))]
    public sealed class ChatterboxEngineEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
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
            ChatterboxModelSetEditor.ShowDownload(set);
        }
    }
}
#endif
