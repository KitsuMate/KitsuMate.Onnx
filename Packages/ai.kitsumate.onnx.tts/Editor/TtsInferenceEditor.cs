using System;
using System.IO;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using KitsuMate.Onnx.Editor;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Editor
{
    public abstract class TtsInferenceEditor : InferenceEngineEditor<TtsRequest, TtsResult>
    {
        [Serializable]
        protected sealed class Inputs
        {
            public string Text = "Hello, this is a local voice test.";
            public string Language = "en";
            public string Reference;
            public string Transcript;
            public string Instruction = "warm, calm voice";
            public int Mode;
            public float Speed = 1;
            public float Duration;
            public float Exaggeration = 0.5f;
            public int MaxTokens = 256;
            public float RepetitionPenalty = 1.2f;
        }
        protected readonly Inputs Input = new();
        private AudioClip clip;
        private string previewAssetPath;
        protected override object TestInputs => Input;
        protected virtual bool SupportsVoiceDesign => false;

        protected override void DrawTestInputs()
        {
            EditorGUILayout.LabelField("Text");
            Input.Text = EditorGUILayout.TextArea(Input.Text, GUILayout.MinHeight(60));
            if (((TtsEngine)Engine).IsMultilingual) Input.Language = EditorGUILayout.TextField("Language", Input.Language);
            if (SupportsVoiceDesign)
            {
                Input.Mode = EditorGUILayout.Popup("Voice", Input.Mode, new[] { "Automatic", "Design", "Clone" });
                if (Input.Mode == 1) Input.Instruction = EditorGUILayout.TextField("Voice instruction", Input.Instruction);
                if (Input.Mode == 2)
                {
                    Input.Reference = InferenceTestPreferences.AssetField<AudioClip>("Reference audio", Input.Reference);
                    Input.Transcript = EditorGUILayout.TextField("Reference transcript", Input.Transcript);
                }
                Input.Speed = EditorGUILayout.FloatField("Speed", Input.Speed);
                Input.Duration = EditorGUILayout.FloatField("Duration seconds (0 = auto)", Input.Duration);
            }
            else
            {
                Input.Reference = InferenceTestPreferences.AssetField<AudioClip>("Reference audio", Input.Reference);
                Input.Exaggeration = EditorGUILayout.FloatField("Exaggeration", Input.Exaggeration);
                Input.MaxTokens = EditorGUILayout.IntField("Max new tokens", Input.MaxTokens);
                Input.RepetitionPenalty = EditorGUILayout.FloatField("Repetition penalty", Input.RepetitionPenalty);
            }
        }

        protected override Func<CancellationToken, Task<TtsRequest>> CaptureRequest()
        {
            if (string.IsNullOrWhiteSpace(Input.Text)) throw new ArgumentException("Enter text to synthesize.");
            var request = new TtsRequest(Input.Text)
            {
                LanguageId = ((TtsEngine)Engine).IsMultilingual ? Input.Language : null,
                VoiceReference = !SupportsVoiceDesign || Input.Mode == 2 ? InferenceTestPreferences.Asset<AudioClip>(Input.Reference) : null,
                VoiceReferenceText = SupportsVoiceDesign && Input.Mode == 2 ? Input.Transcript : null,
                VoiceInstruction = SupportsVoiceDesign && Input.Mode == 1 ? Input.Instruction : null,
                Speed = SupportsVoiceDesign ? Input.Speed : 1,
                DurationSeconds = SupportsVoiceDesign ? Input.Duration : 0,
                Exaggeration = Input.Exaggeration, MaxNewTokens = Input.MaxTokens, RepetitionPenalty = Input.RepetitionPenalty
            };
            if (SupportsVoiceDesign && Input.Mode == 2 && (request.VoiceReference == null || string.IsNullOrWhiteSpace(request.VoiceReferenceText)))
                throw new ArgumentException("Cloning requires reference audio and its transcript.");
            if (SupportsVoiceDesign && Input.Mode == 1 && string.IsNullOrWhiteSpace(request.VoiceInstruction))
                throw new ArgumentException("Enter a voice instruction.");
            if (!float.IsFinite(request.Speed) || request.Speed <= 0 || !float.IsFinite(request.DurationSeconds) || request.DurationSeconds < 0)
                throw new ArgumentException("Speed must be positive and duration must be zero or positive.");
            return _ => Task.FromResult(request);
        }

        protected override void AcceptResult(TtsResult result)
        {
            // AudioUtil requires imported audio; AudioClip.Create produces a silent preview.
            previewAssetPath = $"Assets/InferencePreview-{Guid.NewGuid():N}.wav";
            try
            {
                WriteWav(previewAssetPath, result);
                AssetDatabase.ImportAsset(previewAssetPath, ImportAssetOptions.ForceSynchronousImport);
                clip = AssetDatabase.LoadAssetAtPath<AudioClip>(previewAssetPath);
                if (clip == null) throw new InvalidOperationException("Could not import the generated audio preview.");
            }
            catch
            {
                ClearResult();
                throw;
            }
        }
        protected override void DrawTestResult()
        {
            EditorGUILayout.LabelField("Audio", $"{Result.Duration:F2} seconds, {Result.SampleRate} Hz");
            if (clip == null) return;
            if (GUILayout.Button("Play")) EditorAudioPreview.Play(clip);
            if (GUILayout.Button("Stop")) EditorAudioPreview.Stop();
            if (GUILayout.Button("Save WAV"))
            {
                string path = EditorUtility.SaveFilePanel("Save generated audio", "", "voice", "wav");
                if (!string.IsNullOrEmpty(path)) WriteWav(path, Result);
            }
        }
        protected override void ClearResult()
        {
            if (clip != null) EditorAudioPreview.Stop();
            clip = null;
            if (string.IsNullOrEmpty(previewAssetPath)) return;
            AssetDatabase.DeleteAsset(previewAssetPath);
            previewAssetPath = null;
        }
        public static void WriteWav(string path, TtsResult result)
        {
            using var writer = new BinaryWriter(File.Create(path));
            int size = checked(result.Samples.Length * 2);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); writer.Write(36 + size);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt ")); writer.Write(16);
            writer.Write((short)1); writer.Write((short)1); writer.Write(result.SampleRate);
            writer.Write(result.SampleRate * 2); writer.Write((short)2); writer.Write((short)16);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("data")); writer.Write(size);
            foreach (float sample in result.Samples) writer.Write((short)Mathf.RoundToInt(Mathf.Clamp(sample, -1, 1) * short.MaxValue));
        }
    }

    // Unity exposes Inspector audio preview through an internal Editor API; keep its use isolated.
    internal static class EditorAudioPreview
    {
        private static readonly Type AudioUtil = typeof(AudioImporter).Assembly.GetType("UnityEditor.AudioUtil");
        private static readonly MethodInfo PlayMethod = AudioUtil?.GetMethod("PlayPreviewClip", BindingFlags.Static | BindingFlags.Public,
            null, new[] { typeof(AudioClip), typeof(int), typeof(bool) }, null);
        private static readonly MethodInfo StopMethod = AudioUtil?.GetMethod("StopAllPreviewClips", BindingFlags.Static | BindingFlags.Public);
        public static void Play(AudioClip clip)
        {
            if (PlayMethod == null) { Debug.LogError("Editor audio preview is unavailable in this Unity version. Save WAV to listen."); return; }
            Stop(); PlayMethod.Invoke(null, new object[] { clip, 0, false });
        }
        public static void Stop() => StopMethod?.Invoke(null, null);
    }
}
