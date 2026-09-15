using System;
using System.Threading;
using System.Threading.Tasks;
using KitsuMate.Onnx.Editor;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.LipSync.Editor
{
    public abstract class LipSyncInferenceEditor : InferenceEngineEditor<LipSyncRequest, VisemeTimeline>
    {
        [Serializable] private sealed class Inputs { public string Audio; }
        private readonly Inputs input = new();
        private Vector2 scroll;
        protected override object TestInputs => input;
        protected override void DrawTestInputs() => input.Audio = InferenceTestPreferences.AssetField<AudioClip>("Audio", input.Audio);
        protected override Func<CancellationToken, Task<LipSyncRequest>> CaptureRequest()
        {
            var audio = InferenceTestPreferences.Asset<AudioClip>(input.Audio);
            if (audio == null) throw new ArgumentException("Assign an audio clip.");
            var request = new LipSyncRequest(audio);
            return _ => Task.FromResult(request);
        }
        protected override void DrawTestResult()
        {
            EditorGUILayout.LabelField("Result", $"{Result.FrameCount} visemes, {Result.Duration:F2} seconds");
            scroll = EditorGUILayout.BeginScrollView(scroll, GUILayout.MaxHeight(240));
            foreach (var frame in Result.Frames) EditorGUILayout.LabelField(frame.ToString());
            EditorGUILayout.EndScrollView();
            if (GUILayout.Button("Save JSON")) InferenceTestPreferences.SaveText(JsonUtility.ToJson(Result, true), "json");
        }
    }
}
