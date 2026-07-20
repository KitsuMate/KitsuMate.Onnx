using System;
using System.Threading;
using UnityEngine;
using UnityEngine.Events;

namespace KitsuMate.Onnx.Asr
{
    public sealed class SpeechToText : MonoBehaviour
    {
        [SerializeField] private AsrEngine engine;
        [SerializeField] private OnnxBackend backend;
        [SerializeField] private AudioSource audioSource;
        [SerializeField] private bool autoTranscribe;
        [SerializeField] private bool includeTimestamps = true;
        [SerializeField] private UnityEvent<TranscriptionResult> onTranscription = new();
        [SerializeField] private UnityEvent<string> onTranscriptionText = new();
        [SerializeField] private UnityEvent<string> onError = new();

        private AsrEngineRuntime runtime;
        private CancellationTokenSource lifetime;
        private AudioClip lastClip;
        public AsrEngine Engine { get => engine; set => engine = value; }
        public bool IsTranscribing { get; private set; }
        public bool IsReady => runtime?.IsLoaded == true;
        public UnityEvent<TranscriptionResult> OnTranscription => onTranscription;
        public UnityEvent<string> OnTranscriptionText => onTranscriptionText;
        public UnityEvent<string> OnError => onError;

        private async Awaitable Start()
        {
            lifetime = new CancellationTokenSource();
            if (engine == null || backend == null) return;
            try { runtime = (AsrEngineRuntime)await engine.CreateRuntimeAsync(backend, lifetime.Token); }
            catch (Exception exception) { Report(exception); }
        }

        private void Update()
        {
            if (!autoTranscribe || audioSource == null || audioSource.clip == lastClip) return;
            lastClip = audioSource.clip;
            if (lastClip != null) _ = TranscribeAsync(lastClip);
        }

        public Awaitable<TranscriptionResult> TranscribeAsync(AudioClip clip) => ExecuteAsync(new AsrRequest(clip, includeTimestamps ? AsrOperation.TranscribeWithTimestamps : AsrOperation.Transcribe));
        public Awaitable<TranscriptionResult> ForceAlignAsync(string text, AudioClip clip) => ExecuteAsync(new AsrRequest(clip, AsrOperation.ForceAlign) { ExpectedText = text });

        private async Awaitable<TranscriptionResult> ExecuteAsync(AsrRequest request)
        {
            if (runtime == null) { Report(new InvalidOperationException("ASR runtime is not ready.")); return null; }
            IsTranscribing = true;
            try
            {
                TranscriptionResult result = await runtime.RunAsync(request, lifetime.Token);
                onTranscription.Invoke(result);
                onTranscriptionText.Invoke(result.Text);
                return result;
            }
            catch (Exception exception) { Report(exception); return null; }
            finally { IsTranscribing = false; }
        }

        private void OnDestroy() { lifetime?.Cancel(); runtime?.Dispose(); lifetime?.Dispose(); }
        private void Report(Exception exception) { Debug.LogException(exception, this); onError.Invoke(exception.Message); }
    }
}
