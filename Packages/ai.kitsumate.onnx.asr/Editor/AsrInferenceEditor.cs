using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using KitsuMate.Onnx.Editor;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Asr.Editor
{
    public abstract class AsrInferenceEditor : InferenceEngineEditor<AsrRequest, TranscriptionResult>
    {
        [Serializable] private sealed class Inputs
        {
            public string Audio, Language, ExpectedText;
            public AsrOperation Operation;
            public float Threshold = 0.02f, MinDuration = 0.1f, MaxGap = 0.3f;
        }
        private readonly Inputs input = new();
        private Vector2 scroll;
        protected override object TestInputs => input;
        protected override void DrawTestInputs()
        {
            input.Audio = InferenceTestPreferences.AssetField<AudioClip>("Audio", input.Audio);
            input.Operation = (AsrOperation)EditorGUILayout.EnumPopup("Operation", input.Operation);
            input.Language = EditorGUILayout.TextField("Language (blank = auto)", input.Language);
            if (input.Operation == AsrOperation.ForceAlign || input.Operation == AsrOperation.ForceAlignWithVad)
                input.ExpectedText = EditorGUILayout.TextField("Expected text", input.ExpectedText);
            if (input.Operation == AsrOperation.ForceAlignWithVad)
            {
                input.Threshold = EditorGUILayout.FloatField("VAD volume threshold", input.Threshold);
                input.MinDuration = EditorGUILayout.FloatField("Minimum segment seconds", input.MinDuration);
                input.MaxGap = EditorGUILayout.FloatField("Maximum gap seconds", input.MaxGap);
            }
        }
        protected override Func<CancellationToken, Task<AsrRequest>> CaptureRequest()
        {
            var audio = InferenceTestPreferences.Asset<AudioClip>(input.Audio);
            if (audio == null) throw new ArgumentException("Assign an audio clip.");
            if ((input.Operation == AsrOperation.ForceAlign || input.Operation == AsrOperation.ForceAlignWithVad) && string.IsNullOrWhiteSpace(input.ExpectedText))
                throw new ArgumentException("Alignment requires expected text.");
            var request = new AsrRequest(audio, input.Operation) { Language = input.Language, ExpectedText = input.ExpectedText,
                VadVolumeThreshold = input.Threshold, VadMinSegmentDuration = input.MinDuration, VadMaxSegmentGap = input.MaxGap };
            return _ => Task.FromResult(request);
        }
        public static string FormatResult(TranscriptionResult result) => result.Text + "\n\nLanguage: " + result.Language + "\n" +
            string.Join("\n", result.WordTimestamps.Select(word => word.ToString())) + "\n" +
            string.Join("\n", result.CharacterTimestamps.Select(character => character.ToString()));
        protected override void DrawTestResult()
        {
            string text = FormatResult(Result);
            scroll = EditorGUILayout.BeginScrollView(scroll, GUILayout.MaxHeight(240));
            EditorGUILayout.TextArea(text);
            EditorGUILayout.EndScrollView();
            if (GUILayout.Button("Copy")) EditorGUIUtility.systemCopyBuffer = text;
            if (GUILayout.Button("Save text")) InferenceTestPreferences.SaveText(text, "txt");
        }
    }
}
