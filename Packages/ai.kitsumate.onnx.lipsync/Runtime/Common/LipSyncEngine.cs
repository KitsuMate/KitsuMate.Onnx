using System;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;

namespace KitsuMate.Onnx.LipSync
{
    [Serializable]
    public sealed class LipSyncRequest
    {
        public AudioClip Audio;
        public float[] Samples;
        public int SampleRate;
        public LipSyncRequest(AudioClip audio) { Audio = audio; }
        public LipSyncRequest(float[] samples, int sampleRate) { Samples = samples; SampleRate = sampleRate; }
    }

    public abstract class LipSyncEngine : InferenceEngine<LipSyncRequest, VisemeTimeline> { }

    public abstract class LipSyncEngineRuntime : ThreadedInferenceEngineRuntime<LipSyncRequest, VisemeTimeline>
    {
        public abstract IPhonemeToVisemeMap PhonemeMap { get; }
        protected abstract VisemeTimeline DetectVisemes(float[] samples, int sampleRate);
        protected abstract List<PhonemeFrame> DetectPhonemes(AudioClip audioClip);
        protected sealed override VisemeTimeline OnRun(LipSyncRequest request, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (request == null) throw new ArgumentNullException(nameof(request));
            if (request.Samples != null) return DetectVisemes(request.Samples, request.SampleRate);
            if (request.Audio == null) throw new ArgumentException("Lip sync request requires audio or samples.", nameof(request));
            return DetectVisemesFromPreparedAudio(request.Audio);
        }
        protected abstract VisemeTimeline DetectVisemesFromPreparedAudio(AudioClip audioClip);
    }

    public interface IPhonemeToVisemeMap
    {
        Viseme MapPhoneme(string phoneme);
        IReadOnlyList<string> GetPhonemesForViseme(Viseme viseme);
    }
}
