using System;
using System.Threading;
using UnityEngine;
using UnityEngine.Events;

namespace KitsuMate.Onnx.Tts
{
    public sealed class TextToSpeech : MonoBehaviour
    {
        [SerializeField] private TtsEngine engine;
        [SerializeField] private OnnxBackend backend;
        [SerializeField] private AudioClip voiceReference;
        [SerializeField] private string languageId = "en";
        [SerializeField, Range(0f, 1f)] private float exaggeration = 0.5f;
        [SerializeField] private int maxNewTokens = 256;
        [SerializeField] private float repetitionPenalty = 1.2f;
        [SerializeField] private AudioSource audioSource;
        [SerializeField] private bool autoPlay = true;
        [SerializeField] private UnityEvent<TtsResult> onSynthesis = new();
        [SerializeField] private UnityEvent<AudioClip> onAudioClipReady = new();
        [SerializeField] private UnityEvent<string> onError = new();

        private TtsEngineRuntime runtime;
        private CancellationTokenSource lifetime;
        public TtsEngine Engine { get => engine; set => engine = value; }
        public AudioClip VoiceReference { get => voiceReference; set => voiceReference = value; }
        public bool IsSynthesizing { get; private set; }
        public bool IsReady => runtime?.IsLoaded == true;
        public bool IsModelLoading => runtime?.LoadState == EngineLoadState.Loading;
        public UnityEvent<TtsResult> OnSynthesis => onSynthesis;
        public UnityEvent<AudioClip> OnAudioClipReady => onAudioClipReady;
        public UnityEvent<string> OnError => onError;

        private async Awaitable OnEnable()
        {
            lifetime = new CancellationTokenSource();
            if (engine == null || backend == null) return;
            try { runtime = (TtsEngineRuntime)await engine.CreateRuntimeAsync(backend, lifetime.Token); }
            catch (Exception exception) { Report(exception); }
        }

        public async Awaitable<TtsResult> SynthesizeAsync(string text)
        {
            if (runtime == null) { Report(new InvalidOperationException("TTS runtime is not ready.")); return null; }
            if (string.IsNullOrWhiteSpace(text)) { Report(new ArgumentException("Text is empty.", nameof(text))); return null; }
            IsSynthesizing = true;
            try
            {
                var request = new TtsRequest { Text = text, VoiceReference = voiceReference, LanguageId = languageId, Exaggeration = exaggeration, MaxNewTokens = maxNewTokens, RepetitionPenalty = repetitionPenalty };
                TtsResult result = await runtime.RunAsync(request, lifetime.Token);
                onSynthesis.Invoke(result);
                AudioClip clip = result.ToAudioClip("tts_output");
                onAudioClipReady.Invoke(clip);
                if (autoPlay && audioSource != null) { audioSource.clip = clip; audioSource.Play(); }
                return result;
            }
            catch (Exception exception) { Report(exception); return null; }
            finally { IsSynthesizing = false; }
        }

        private void OnDisable() { lifetime?.Cancel(); runtime?.Dispose(); runtime = null; lifetime?.Dispose(); lifetime = null; }
        private void Report(Exception exception) { Debug.LogException(exception, this); onError.Invoke(exception.Message); }
    }
}
