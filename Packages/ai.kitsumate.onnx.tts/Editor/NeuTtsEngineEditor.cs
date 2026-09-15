#if UNITY_EDITOR
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using KitsuMate.Onnx.Editor.Download;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.NeuTts.Editor
{
    [CustomEditor(typeof(NeuTtsEngine))]
    public sealed class NeuTtsEngineEditor : KitsuMate.Onnx.Tts.Editor.TtsInferenceEditor
    {
        private readonly string[] speakers = { "emily", "paul", "sophie", "steven" };
        private readonly string[] emotions = { "angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised" };
        private int speaker, emotion = 4;
        protected override void DrawEngineInspector()
        {
            DrawDefaultInspector();
            if (GUILayout.Button("Set Up Models")) SetUpModels();
        }

        private void SetUpModels()
        {
            serializedObject.Update();
            var property = serializedObject.FindProperty("modelSet");
            string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(target))?.Replace('\\', '/') ?? "Assets";
            var set = property.objectReferenceValue as NeuTtsModelSet ??
                AssetDatabase.LoadAssetAtPath<NeuTtsModelSet>(directory + "/NeuTtsModelSet.asset");
            if (set == null)
            {
                set = CreateInstance<NeuTtsModelSet>();
                AssetDatabase.CreateAsset(set, AssetDatabase.GenerateUniqueAssetPath(directory + "/NeuTtsModelSet.asset"));
            }
            property.objectReferenceValue = set;
            var backend = serializedObject.FindProperty("backend");
            if (backend.objectReferenceValue == null) backend.objectReferenceValue = OnnxSettings.Load()?.GetBackendForCurrentPlatform();
            serializedObject.ApplyModifiedProperties();
            AssetDatabase.SaveAssets();
            ModelDownloadWindow.Show(set);
        }
        protected override void DrawTestInputs()
        {
            EditorGUILayout.LabelField("Text");
            Input.Text = EditorGUILayout.TextArea(Input.Text, GUILayout.MinHeight(60));
            speaker = EditorGUILayout.Popup("Speaker", speaker, speakers);
            emotion = EditorGUILayout.Popup("Emotion", emotion, emotions);
            Input.MaxTokens = EditorGUILayout.IntField("Max new tokens", Input.MaxTokens);
        }
        protected override Func<CancellationToken, Task<TtsRequest>> CaptureRequest()
        {
            var config = ((NeuTtsEngine)Engine).Generation.Clone();
            config.Speaker = speakers[speaker]; config.Emotion = emotions[emotion];
            var request = new TtsRequest(Input.Text) { NeuTts = config, MaxNewTokens = Input.MaxTokens };
            return _ => Task.FromResult(request);
        }
    }
}
#endif
