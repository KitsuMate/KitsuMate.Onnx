using System;
using System.Threading;
using UnityEngine;

namespace KitsuMate.Onnx.Asr
{
    public enum AsrOperation { Transcribe, TranscribeWithTimestamps, ForceAlign, ForceAlignWithVad, DetectLanguage }

    [Serializable]
    public sealed class AsrRequest
    {
        public AudioClip Audio;
        public AsrOperation Operation;
        public string ExpectedText;
        public string Language;
        public float VadVolumeThreshold = 0.02f;
        public float VadMinSegmentDuration = 0.1f;
        public float VadMaxSegmentGap = 0.3f;
        public AsrRequest(AudioClip audio, AsrOperation operation = AsrOperation.Transcribe) { Audio = audio; Operation = operation; }
    }

    public abstract class AsrEngine : InferenceEngine<AsrRequest, TranscriptionResult> { }

    public abstract class AsrEngineRuntime : ThreadedInferenceEngineRuntime<AsrRequest, TranscriptionResult>
    {
        public abstract System.Collections.Generic.IReadOnlyList<string> SupportedLanguages { get; }
        protected abstract TranscriptionResult Transcribe(AudioClip audioClip);
        protected abstract TranscriptionResult TranscribeWithTimestamps(AudioClip audioClip);
        protected abstract TranscriptionResult ForceAlign(string text, AudioClip audioClip);
        protected virtual TranscriptionResult ForceAlignWithVad(string text, AudioClip audioClip, float volumeThreshold, float minSegmentDuration, float maxSegmentGap) => ForceAlign(text, audioClip);
        protected virtual string DetectLanguage(AudioClip audioClip) => Transcribe(audioClip).Language;
        protected virtual void ApplyRequest(AsrRequest request) { }
        protected sealed override TranscriptionResult OnRun(AsrRequest request, CancellationToken cancellationToken)
        {
            if (request?.Audio == null) throw new ArgumentException("ASR request requires audio.", nameof(request));
            cancellationToken.ThrowIfCancellationRequested();
            ApplyRequest(request);
            return request.Operation switch
            {
                AsrOperation.Transcribe => Transcribe(request.Audio),
                AsrOperation.TranscribeWithTimestamps => TranscribeWithTimestamps(request.Audio),
                AsrOperation.ForceAlign => ForceAlign(request.ExpectedText, request.Audio),
                AsrOperation.ForceAlignWithVad => ForceAlignWithVad(request.ExpectedText, request.Audio, request.VadVolumeThreshold, request.VadMinSegmentDuration, request.VadMaxSegmentGap),
                AsrOperation.DetectLanguage => new TranscriptionResult(string.Empty, DetectLanguage(request.Audio)),
                _ => throw new ArgumentOutOfRangeException()
            };
        }
    }
}
