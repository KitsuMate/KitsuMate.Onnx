#if UNITY_EDITOR
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Editor.Download;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Chatterbox.Editor
{
    [CustomEditor(typeof(ChatterboxModelSet))]
    public sealed class ChatterboxModelSetEditor : UnityEditor.Editor
    {
        private static readonly (string Label, string Repository, string Revision)[] Downloads =
        {
            ("Chatterbox Turbo", "KitsuMate/chatterbox-turbo-onnx",
                "7320dc3cfac446f5d689565d1f701fd1a0b8e516"),
            ("Chatterbox Nano", "KitsuMate/chatterbox-nano-onnx",
                "b70ba9ceb90a146e93af372d805ec76baaa48b0b"),
            ("Chatterbox", "KitsuMate/chatterbox-onnx",
                "ed9d6008739dc30ff4944973944b7b3d973e054f"),
            ("Chatterbox Multilingual", "KitsuMate/chatterbox-multilingual-ONNX",
                "d218e7492aa91033aec45a2f5234ff0cf1afff82")
        };

        private int downloadIndex;

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
            downloadIndex = EditorGUILayout.Popup("Model", downloadIndex,
                Downloads.Select(download => download.Label).ToArray());
            if (GUILayout.Button("Download Models"))
                ShowDownload(set, downloadIndex);
            ModelSetEditorUi.Validation(set);
        }

        internal static void ShowDownload(ChatterboxModelSet set)
        {
            ShowDownload(set, 0);
        }

        private static void ShowDownload(ChatterboxModelSet set, int index)
        {
            var download = Downloads[Mathf.Clamp(index, 0, Downloads.Length - 1)];
            ModelDownloadWindow.Show(new ModelDownloadRequest(download.Repository, download.Revision, "chatterbox"),
                result =>
            {
                result.RequireGraph("speech-encoder", new[] { "audio_values" },
                    new[] { "audio_features", "audio_tokens", "speaker_embeddings", "speaker_features" });
                result.RequireGraph("embed-tokens", new[] { "input_ids" });
                result.RequireGraph("language-model", new[] { "inputs_embeds", "attention_mask" },
                    new[] { "logits" });
                result.RequireGraph("conditional-decoder",
                    new[] { "speech_tokens", "speaker_embeddings", "speaker_features" },
                    new[] { "waveform" });
                int cacheInputCount = result.GetInputNames("language-model")
                    .Count(name => name.StartsWith("past_key_values.", System.StringComparison.Ordinal));
                int cacheOutputCount = result.GetOutputNames("language-model")
                    .Count(name => name.StartsWith("present.", System.StringComparison.Ordinal));
                if (cacheInputCount == 0 || cacheInputCount != cacheOutputCount)
                    throw new System.InvalidOperationException(
                        "Chatterbox language model cache inputs and outputs do not match.");
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
