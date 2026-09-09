#if UNITY_EDITOR
using System.IO;
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Editor.Download;
using KitsuMate.Onnx.Download;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.OmniVoice.Editor
{
    [CustomEditor(typeof(OmniVoiceModelSet))]
    public sealed class OmniVoiceModelSetEditor : UnityEditor.Editor
    {
        internal const string Repository = "KitsuMate/omnivoice-onnx";
        internal const string Revision = "45d20c87b64f35c4ac203c5bac7ad97a3e60ca95";

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            var set = (OmniVoiceModelSet)target;
            ModelSetEditorUi.Header(set, "OmniVoice Model Set");
            EditorGUILayout.PropertyField(serializedObject.FindProperty("topology"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("codecPrecision"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("profile"));
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Backbone", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("mergedBackbone"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("audioEmbeddingsEncoder"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("languageDecoder"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("audioHeadsDecoder"));
            EditorGUILayout.LabelField("Higgs Codec", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("acousticEncoder"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("semanticEncoder"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("quantizerEncoder"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("higgsDecoder"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("tokenizer"));
            serializedObject.ApplyModifiedProperties();
            if (GUILayout.Button("Download CPU Compact INT4 Profile"))
                Download(set, "omnivoice-cpu", "CPU compact INT4");
            if (GUILayout.Button("Download Portable FP32 Profile"))
                Download(set, "omnivoice-portable", "Portable FP32");
            ModelSetEditorUi.Validation(set);
        }

        private static void Download(OmniVoiceModelSet set, string family, string profile)
        {
            ModelDownloadWindow.Show(new ModelDownloadRequest(Repository, Revision, family), result =>
            {
                result.RequireGraph("merged-backbone",
                    new[] { "input_ids", "audio_mask", "attention_mask", "position_ids" }, new[] { "logits" });
                result.ConfigureModel(set.MergedBackbone, "merged-backbone");
                result.RequireGraph("acoustic-encoder", new[] { "waveform_24k" }, new[] { "acoustic_features" });
                result.RequireGraph("semantic-encoder", new[] { "waveform_16k" }, new[] { "semantic_features" });
                result.RequireGraph("quantizer-encoder", new[] { "acoustic_features", "semantic_features" }, new[] { "codes" });
                result.RequireGraph("higgs-decoder", new[] { "codes" }, new[] { "waveform_24k" });
                result.ConfigureModel(set.AcousticEncoder, "acoustic-encoder");
                result.ConfigureModel(set.SemanticEncoder, "semantic-encoder");
                result.ConfigureModel(set.QuantizerEncoder, "quantizer-encoder");
                result.ConfigureModel(set.HiggsDecoder, "higgs-decoder");
                set.Configure(OmniVoiceBackboneTopology.Merged, OmniVoiceTensorPrecision.Float32, profile,
                    result.LoadAsset<TextAsset>("tokenizer"));
                result.ApplyMetadata(set);
                AssetDatabase.SaveAssets();
                Selection.activeObject = set;
            });
        }
    }

    [CustomEditor(typeof(OmniVoiceEngine))]
    public sealed class OmniVoiceEngineEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
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
