using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Events;

namespace KitsuMate.Onnx.LipSync
{
    /// <summary>
    /// Component that plays viseme animations based on lip sync data.
    /// Supports real-time audio analysis or pre-generated timelines.
    /// Discovers all SkinnedMeshRenderers under the target root and applies
    /// blend shapes wherever they exist, driven by a VisemeBlendShapeProfile.
    /// </summary>
    [AddComponentMenu("KitsuMate/ONNX/Lip Sync Player")]
    public class LipSyncPlayer : MonoBehaviour
    {
        /// <summary>
        /// Mode for lip sync playback.
        /// </summary>
        public enum PlaybackMode
        {
            /// <summary>Pre-analyzed timeline synchronized to audio.</summary>
            Timeline,
            /// <summary>Real-time analysis of audio (higher latency).</summary>
            Realtime
        }
        
        [Header("Engine")]
        [SerializeField, Tooltip("Lip sync engine for detection")]
        private LipSyncEngine _engine;
        [SerializeField, Tooltip("Caller-owned ONNX backend")]
        private OnnxBackend _backend;
        
        [Header("Audio")]
        [SerializeField, Tooltip("Audio source to sync with")]
        private AudioSource _audioSource;
        
        [SerializeField, Tooltip("Playback mode")]
        private PlaybackMode _mode = PlaybackMode.Timeline;

        [SerializeField, Min(0f), Tooltip("Delay in seconds before playback begins when using the delayed play helpers")]
        private float _playDelay;

        [SerializeField, Range(0.15f, 1f), Tooltip("Audio window length used for realtime detection")]
        private float _realtimeWindowSeconds = 0.35f;

        [SerializeField, Range(0.02f, 0.25f), Tooltip("How often realtime detection is triggered while audio is playing")]
        private float _realtimeAnalysisInterval = 0.08f;

        [SerializeField, Range(0f, 0.12f), Tooltip("Bias toward the most recent audio when selecting a realtime viseme")]
        private float _realtimeLookbackSeconds = 0.03f;

        [SerializeField, Range(0f, 0.1f), Tooltip("Minimum RMS required before realtime analysis updates visemes")]
        private float _realtimeMinRms = 0.01f;
        
        [Header("Target")]
        [SerializeField, Tooltip("Root GameObject to search for SkinnedMeshRenderers. If null, uses this GameObject.")]
        private GameObject _targetRoot;
        
        [SerializeField, Tooltip("Include inactive SkinnedMeshRenderers in the search")]
        private bool _includeInactive;
        
        [Header("Blend Shape Profile")]
        [SerializeField, Tooltip("Profile that maps visemes to blend shape names and weights")]
        private VisemeBlendShapeProfile _profile;
        
        [Header("Animation")]
        [SerializeField, Range(0f, 1f), Tooltip("Maximum blend shape weight")]
        private float _maxWeight = 1f;
        
        [SerializeField, Range(0.01f, 0.5f), Tooltip("Smoothing time for transitions")]
        private float _smoothTime = 0.05f;
        
        [SerializeField, Tooltip("Time offset in seconds (negative = earlier, positive = later)")]
        private float _timeOffset = 0f;
        
        [Header("Events")]
        [SerializeField]
        private UnityEvent<VisemeFrame> _onVisemeChanged;
        
        // Runtime state
        private VisemeTimeline _timeline;
        private AudioClip _timelineClip;
        private List<ResolvedBlendShapeTarget> _resolvedTargets;
        private float[] _currentWeights;
        private float[] _targetWeights;
        private float[] _velocities;
        private Viseme _currentViseme;
        private CancellationTokenSource _analysisCts;
        private LipSyncEngineRuntime _runtime;
        private Task _modelLoadTask;
        private Task _realtimeAnalysisTask;
        private bool _isAnalyzing;
        private bool _wasAudioPlaying;
        private int _playbackGeneration;
        private int _timelineGeneration;
        private PlaybackRequest? _pendingPlaybackRequest;
        private AudioSourceSampleTap _realtimeSampleTap;
        private float[] _realtimeAnalysisBuffer;
        private float _nextRealtimeAnalysisAt;
        private readonly object _realtimeResultLock = new();
        private RealtimeAnalysisResult? _pendingRealtimeResult;

        private readonly TimelinePlaybackDriver _timelineDriver = new();

        private struct PlaybackRequest
        {
            public AudioClip Clip;
            public int Generation;
        }

        private struct RealtimeAnalysisResult
        {
            public int Generation;
            public VisemeFrame? Frame;
        }
        
        private struct ResolvedBlendShapeTarget
        {
            public SkinnedMeshRenderer Renderer;
            public int BlendShapeIndex;
            public float TargetWeight;
            public int VisemeIndex;
        }
        
        /// <summary>LipSync engine for detection.</summary>
        public LipSyncEngine Engine
        {
            get => _engine;
            set => _engine = value;
        }
        
        /// <summary>Audio source to sync with.</summary>
        public AudioSource AudioSource
        {
            get => _audioSource;
            set
            {
                if (_audioSource == value)
                    return;

                _audioSource = value;
                _wasAudioPlaying = _audioSource != null && _audioSource.isPlaying;
            }
        }

        /// <summary>Playback mode for lip sync updates.</summary>
        public PlaybackMode Mode
        {
            get => _mode;
            set => _mode = value;
        }
        
        /// <summary>Root GameObject to search for SkinnedMeshRenderers.</summary>
        public GameObject TargetRoot
        {
            get => _targetRoot;
            set => _targetRoot = value;
        }
        
        /// <summary>Viseme-to-blend-shape profile.</summary>
        public VisemeBlendShapeProfile Profile
        {
            get => _profile;
            set => _profile = value;
        }
        
        /// <summary>Current viseme timeline.</summary>
        public VisemeTimeline Timeline
        {
            get => _timeline;
            set => _timeline = value;
        }
        
        /// <summary>Maximum blend shape weight.</summary>
        public float MaxWeight
        {
            get => _maxWeight;
            set => _maxWeight = Mathf.Clamp01(value);
        }
        
        /// <summary>Smoothing time for transitions.</summary>
        public float SmoothTime
        {
            get => _smoothTime;
            set => _smoothTime = Mathf.Clamp(value, 0.01f, 0.5f);
        }
        
        /// <summary>Time offset for synchronization.</summary>
        public float TimeOffset
        {
            get => _timeOffset;
            set => _timeOffset = value;
        }

        /// <summary>Configured delay for the delayed playback helper.</summary>
        public float PlayDelay
        {
            get => _playDelay;
            set => _playDelay = Mathf.Max(0f, value);
        }

        /// <summary>Whether realtime mode is actively following a playing AudioSource.</summary>
        public bool IsRealtimeActive => _mode == PlaybackMode.Realtime && _audioSource != null && _audioSource.isPlaying;
        
        /// <summary>Event fired when viseme changes.</summary>
        public UnityEvent<VisemeFrame> OnVisemeChanged => _onVisemeChanged;
        
        /// <summary>Current active viseme.</summary>
        public Viseme CurrentViseme => _currentViseme;
        
        /// <summary>Whether analysis is in progress.</summary>
        public bool IsAnalyzing => _isAnalyzing;

        /// <summary>Whether the assigned engine is currently loading.</summary>
        public bool IsModelLoading => _runtime?.LoadState == EngineLoadState.Loading;
        
        private void Awake()
        {
            int visemeCount = Enum.GetValues(typeof(Viseme)).Length;
            _currentWeights = new float[visemeCount];
            _targetWeights = new float[visemeCount];
            _velocities = new float[visemeCount];
            _wasAudioPlaying = _audioSource != null && _audioSource.isPlaying;
            
            Refresh();
        }

        private async void OnEnable()
        {
            if (_engine == null || _backend == null || _runtime != null) return;
            try { _runtime = (LipSyncEngineRuntime)await _engine.CreateRuntimeAsync(_backend); }
            catch (Exception exception) { Debug.LogException(exception, this); }
        }
        

        
        private void OnDisable()
        {
            CancelAnalysis();
            _runtime?.Dispose();
            _runtime = null;
            _modelLoadTask = null;
            _realtimeAnalysisTask = null;
            _timelineDriver.Reset();
            _pendingPlaybackRequest = null;
            _wasAudioPlaying = false;
            ResetRealtimeState();
        }
        
        private void Update()
        {
            UpdatePlaybackState();
            UpdateVisemes();
            
            ApplyBlendShapes();
        }

        private void UpdatePlaybackState()
        {
            bool isPlaying = _audioSource != null && _audioSource.isPlaying;

            if (isPlaying && !_wasAudioPlaying)
            {
                OnPlaybackStarted();
            }
            else if (!isPlaying && _wasAudioPlaying)
            {
                OnPlaybackStopped();
            }

            _wasAudioPlaying = isPlaying;
        }

        private void OnPlaybackStarted()
        {
            int generation = _pendingPlaybackRequest?.Generation ?? ++_playbackGeneration;
            _playbackGeneration = generation;
            _timelineDriver.BeginPlayback(generation);
            _pendingPlaybackRequest = null;

            if (_mode == PlaybackMode.Timeline)
            {
                StartTimelineAnalysisIfNeeded(generation);
            }

            if (_mode == PlaybackMode.Realtime)
            {
                ResetVisemes();
                EnsureRealtimeTap();
                _nextRealtimeAnalysisAt = 0f;
            }
        }

        private void OnPlaybackStopped()
        {
            _timelineDriver.Reset();
            if (_mode == PlaybackMode.Realtime)
            {
                ResetRealtimeState();
                ResetVisemes();
            }
        }

        private void UpdateVisemes()
        {
            if (_audioSource == null || !_audioSource.isPlaying)
                return;

            switch (_mode)
            {
                case PlaybackMode.Timeline:
                    UpdateVisemesFromTimeline();
                    break;
                case PlaybackMode.Realtime:
                    UpdateVisemesRealtime();
                    break;
            }
        }

        private void UpdateVisemesRealtime()
        {
            ApplyPendingRealtimeResult();

            if (_runtime == null)
            {
                SetVisemeSilence();
                return;
            }

            if (!_runtime.IsLoaded)
            {
                EnsureModelPreloadStarted();
                SetVisemeSilence();
                return;
            }

            var tap = EnsureRealtimeTap();
            if (tap == null)
            {
                SetVisemeSilence();
                return;
            }

            if (_realtimeAnalysisTask != null && !_realtimeAnalysisTask.IsCompleted)
                return;

            if (Time.unscaledTime < _nextRealtimeAnalysisAt)
                return;

            int sampleRate = tap.SampleRate;
            if (sampleRate <= 0)
                return;

            int sampleCount = Mathf.CeilToInt((_realtimeWindowSeconds + _realtimeLookbackSeconds) * sampleRate);
            EnsureRealtimeAnalysisBuffer(sampleCount);

            if (!tap.TryCopyLatestWindow(_realtimeAnalysisBuffer, sampleCount, out sampleRate, out float rms))
                return;

            _nextRealtimeAnalysisAt = Time.unscaledTime + _realtimeAnalysisInterval;

            if (rms < _realtimeMinRms)
            {
                QueueRealtimeResult(new RealtimeAnalysisResult
                {
                    Generation = _playbackGeneration,
                    Frame = null
                });
                return;
            }

            float[] samples = new float[sampleCount];
            Array.Copy(_realtimeAnalysisBuffer, samples, sampleCount);

            _isAnalyzing = true;
            int generation = _playbackGeneration;
            _realtimeAnalysisTask = AnalyzeRealtimeWindowAsync(samples, sampleRate, generation);
        }
        
        /// <summary>
        /// Re-discover SkinnedMeshRenderers and resolve blend shape indices.
        /// Call after changing the target root, profile, or after the hierarchy changes at runtime.
        /// </summary>
        public void Refresh()
        {
            _resolvedTargets = new List<ResolvedBlendShapeTarget>();
            
            if (_profile == null)
                return;
            
            var root = _targetRoot != null ? _targetRoot : gameObject;
            var renderers = root.GetComponentsInChildren<SkinnedMeshRenderer>(_includeInactive);
            
            var visemes = (Viseme[])Enum.GetValues(typeof(Viseme));
            
            foreach (var renderer in renderers)
            {
                if (renderer.sharedMesh == null)
                    continue;
                
                var mesh = renderer.sharedMesh;
                
                foreach (var viseme in visemes)
                {
                    var targets = _profile.GetTargetsForViseme(viseme);
                    
                    foreach (var target in targets)
                    {
                        int index = VisemeBlendShapeProfile.ResolveBlendShapeIndex(mesh, target.BlendShapeName);
                        if (index >= 0)
                        {
                            _resolvedTargets.Add(new ResolvedBlendShapeTarget
                            {
                                Renderer = renderer,
                                BlendShapeIndex = index,
                                TargetWeight = target.Weight,
                                VisemeIndex = (int)viseme
                            });
                        }
                    }
                }
            }
        }
        
        /// <summary>
        /// Analyze an audio clip and store the timeline.
        /// </summary>
        public async Awaitable AnalyzeClipAsync(AudioClip clip, CancellationToken cancellationToken = default)
        {
            if (_engine == null || _backend == null)
                throw new InvalidOperationException("LipSyncEngine is not assigned");
            if (_runtime == null) _runtime = (LipSyncEngineRuntime)await _engine.CreateRuntimeAsync(_backend, cancellationToken);
            
            CancelAnalysis(invalidatePlayback: false);
            _analysisCts = new CancellationTokenSource();
            int generation = _playbackGeneration;
            
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
                cancellationToken, _analysisCts.Token);
            
            _isAnalyzing = true;
            try
            {
                var timeline = await _runtime.RunAsync(new LipSyncRequest(clip), linkedCts.Token);
                if (cancellationToken.IsCancellationRequested || generation != _playbackGeneration)
                    return;

                _timeline = timeline;
                _timelineClip = clip;
                _timelineGeneration = generation;
                _timelineDriver.NotifyTimelineReady(generation);
            }
            finally
            {
                _isAnalyzing = false;
            }
        }

        private async Task PreloadModelAsync()
        {
            if (_runtime == null) _runtime = (LipSyncEngineRuntime)await _engine.CreateRuntimeAsync(_backend);
        }

        private void EnsureModelPreloadStarted()
        {
            if (_engine == null || _backend == null || _runtime?.IsLoaded == true)
                return;

            if (_modelLoadTask != null && !_modelLoadTask.IsCompleted)
                return;
            _modelLoadTask = PreloadModelAsync();
            ObserveModelLoadTask(_modelLoadTask);
        }

        private static void ObserveModelLoadTask(Task modelLoadTask)
        {
            _ = modelLoadTask.ContinueWith(
                task => _ = task.Exception,
                CancellationToken.None,
                TaskContinuationOptions.OnlyOnFaulted,
                TaskScheduler.Default);
        }
        
        /// <summary>
        /// Play audio with synchronized lip sync.
        /// If timeline is not yet analyzed, will analyze first.
        /// </summary>
        public async Awaitable PlayAsync(AudioClip clip, CancellationToken cancellationToken = default)
        {
            if (_audioSource == null)
                throw new InvalidOperationException("AudioSource is not assigned");

            int generation = RegisterPlaybackRequest(clip);
            _audioSource.clip = clip;
            if (NeedsTimelineAnalysis(clip))
            {
                await AnalyzeClipAsync(clip, cancellationToken);
            }
            else
            {
                _timelineGeneration = generation;
                _timelineDriver.NotifyTimelineReady(generation);
            }

            _audioSource.Play();
        }

        /// <summary>
        /// Play audio with synchronized lip sync after a delay.
        /// Timeline analysis runs during the delay window so playback can begin in sync.
        /// </summary>
        public async Awaitable PlayDelayedAsync(AudioClip clip, float delaySeconds, CancellationToken cancellationToken = default)
        {
            if (_audioSource == null)
                throw new InvalidOperationException("AudioSource is not assigned");

            int generation = RegisterPlaybackRequest(clip);
            _audioSource.clip = clip;

            if (NeedsTimelineAnalysis(clip))
            {
                await AnalyzeClipAsync(clip, cancellationToken);
            }
            else
            {
                _timelineGeneration = generation;
                _timelineDriver.NotifyTimelineReady(generation);
            }

            _audioSource.PlayDelayed(Mathf.Max(0f, delaySeconds));
        }

        /// <summary>
        /// Play audio with the configured delayed start.
        /// </summary>
        public Awaitable PlayDelayedAsync(AudioClip clip, CancellationToken cancellationToken = default)
        {
            return PlayDelayedAsync(clip, _playDelay, cancellationToken);
        }
        
        /// <summary>
        /// Stop playback and reset visemes.
        /// </summary>
        public void Stop()
        {
            if (_audioSource != null)
            {
                _audioSource.Stop();
            }
            
            CancelAnalysis();
            _pendingPlaybackRequest = null;
            _timelineDriver.Reset();
            ResetVisemes();
        }
        
        /// <summary>
        /// Cancel any ongoing analysis.
        /// </summary>
        public void CancelAnalysis()
        {
            CancelAnalysis(invalidatePlayback: true);
        }

        private void CancelAnalysis(bool invalidatePlayback)
        {
            if (invalidatePlayback)
                _playbackGeneration++;

            _analysisCts?.Cancel();
            _analysisCts?.Dispose();
            _analysisCts = null;
            _isAnalyzing = false;
            ResetRealtimeState();
        }
        
        /// <summary>
        /// Reset all viseme weights to zero.
        /// </summary>
        public void ResetVisemes()
        {
            for (int i = 0; i < _targetWeights.Length; i++)
            {
                _targetWeights[i] = 0;
            }
            _currentViseme = Viseme.sil;
        }
        
        /// <summary>
        /// Set the current viseme directly (for manual control).
        /// </summary>
        public void SetViseme(Viseme viseme, float weight = 1f)
        {
            for (int i = 0; i < _targetWeights.Length; i++)
            {
                _targetWeights[i] = 0;
            }
            
            _targetWeights[(int)viseme] = weight * _maxWeight;
            
            if (_currentViseme != viseme)
            {
                _currentViseme = viseme;
                _onVisemeChanged?.Invoke(new VisemeFrame(viseme, 0, 0, weight));
            }
        }
        
        private void UpdateVisemesFromTimeline()
        {
            if (_timeline == null || _timelineGeneration != _playbackGeneration)
            {
                SetVisemeSilence();
                return;
            }

            float time = GetCurrentAudioTimeSeconds();
            var frame = _timelineDriver.Sample(_timeline, time);
            ApplyTimelineFrame(frame);
        }

        private float GetCurrentAudioTimeSeconds()
        {
            if (_audioSource == null || _audioSource.clip == null || _audioSource.clip.frequency <= 0)
                return _timeOffset;

            float timeFromSamples = (float)_audioSource.timeSamples / _audioSource.clip.frequency;
            return Mathf.Max(0f, timeFromSamples + _timeOffset);
        }

        private void ApplyTimelineFrame(VisemeFrame? frame)
        {
            for (int i = 0; i < _targetWeights.Length; i++)
                _targetWeights[i] = 0f;

            if (!frame.HasValue)
            {
                _currentViseme = Viseme.sil;
                return;
            }

            var value = frame.Value;
            _targetWeights[(int)value.Viseme] = value.Weight * _maxWeight;

            if (_currentViseme != value.Viseme)
            {
                _currentViseme = value.Viseme;
                _onVisemeChanged?.Invoke(value);
            }
        }

        private void SetVisemeSilence()
        {
            for (int i = 0; i < _targetWeights.Length; i++)
                _targetWeights[i] = 0f;

            _currentViseme = Viseme.sil;
        }

        private AudioSourceSampleTap EnsureRealtimeTap()
        {
            if (_audioSource == null)
                return null;

            if (_realtimeSampleTap == null || _realtimeSampleTap.gameObject != _audioSource.gameObject)
            {
                _realtimeSampleTap = _audioSource.GetComponent<AudioSourceSampleTap>();
                if (_realtimeSampleTap == null)
                    _realtimeSampleTap = _audioSource.gameObject.AddComponent<AudioSourceSampleTap>();
            }

            int sampleRate = AudioSettings.outputSampleRate;
            int capacitySamples = Mathf.CeilToInt((_realtimeWindowSeconds + _realtimeLookbackSeconds + _realtimeAnalysisInterval) * sampleRate);
            _realtimeSampleTap.Configure(Mathf.Max(capacitySamples, 1));
            return _realtimeSampleTap;
        }

        private void EnsureRealtimeAnalysisBuffer(int sampleCount)
        {
            if (_realtimeAnalysisBuffer == null || _realtimeAnalysisBuffer.Length != sampleCount)
                _realtimeAnalysisBuffer = new float[sampleCount];
        }

        private async Task AnalyzeRealtimeWindowAsync(float[] samples, int sampleRate, int generation)
        {
            try
            {
                var timeline = await _runtime.RunAsync(new LipSyncRequest(samples, sampleRate));
                if (generation != _playbackGeneration)
                    return;

                QueueRealtimeResult(new RealtimeAnalysisResult
                {
                    Generation = generation,
                    Frame = SelectRealtimeFrame(timeline)
                });
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
            }
            finally
            {
                if (generation == _playbackGeneration)
                    _isAnalyzing = false;
            }
        }

        private VisemeFrame? SelectRealtimeFrame(VisemeTimeline timeline)
        {
            if (timeline == null || timeline.FrameCount == 0)
                return null;

            float sampleTime = Mathf.Max(0f, timeline.Duration - _realtimeLookbackSeconds);
            var frame = timeline.GetFrameAtTime(sampleTime);
            if (frame.HasValue)
                return frame;

            var frames = timeline.Frames;
            return frames.Count > 0 ? frames[frames.Count - 1] : null;
        }

        private void QueueRealtimeResult(RealtimeAnalysisResult result)
        {
            lock (_realtimeResultLock)
            {
                _pendingRealtimeResult = result;
            }
        }

        private void ApplyPendingRealtimeResult()
        {
            RealtimeAnalysisResult? result = null;
            lock (_realtimeResultLock)
            {
                if (_pendingRealtimeResult.HasValue)
                {
                    result = _pendingRealtimeResult;
                    _pendingRealtimeResult = null;
                }
            }

            if (!result.HasValue || result.Value.Generation != _playbackGeneration)
                return;

            ApplyTimelineFrame(result.Value.Frame);
        }

        private void ResetRealtimeState()
        {
            lock (_realtimeResultLock)
            {
                _pendingRealtimeResult = null;
            }

            _realtimeAnalysisTask = null;
            _nextRealtimeAnalysisAt = 0f;
            _realtimeAnalysisBuffer = null;
            _realtimeSampleTap?.Clear();
        }

        private int RegisterPlaybackRequest(AudioClip clip)
        {
            int generation = ++_playbackGeneration;
            _timelineDriver.Reset();
            ResetRealtimeState();
            _pendingPlaybackRequest = new PlaybackRequest
            {
                Clip = clip,
                Generation = generation
            };

            return generation;
        }

        private bool NeedsTimelineAnalysis(AudioClip clip)
        {
            return _timeline == null || _timelineClip != clip;
        }

        private void StartTimelineAnalysisIfNeeded(int generation)
        {
            if (_audioSource == null || _audioSource.clip == null)
                return;

            if (!NeedsTimelineAnalysis(_audioSource.clip))
            {
                _timelineGeneration = generation;
                _timelineDriver.NotifyTimelineReady(generation);
                return;
            }

            if (_isAnalyzing)
                return;

            _ = AnalyzeTimelineForCurrentClipAsync(generation, _audioSource.clip);
        }

        private async Awaitable AnalyzeTimelineForCurrentClipAsync(int generation, AudioClip clip)
        {
            try
            {
                await AnalyzeClipAsync(clip);
            }
            catch (Exception exception)
            {
                if (generation == _playbackGeneration)
                    Debug.LogException(exception);
            }
        }

        private sealed class TimelinePlaybackDriver
        {
            private int _activeGeneration;
            private bool _hasTimeline;

            public void BeginPlayback(int generation)
            {
                _activeGeneration = generation;
            }

            public void NotifyTimelineReady(int generation)
            {
                if (_activeGeneration == generation)
                    _hasTimeline = true;
            }

            public VisemeFrame? Sample(VisemeTimeline timeline, float timeSeconds)
            {
                if (!_hasTimeline)
                    return null;

                return timeline.GetFrameAtTime(timeSeconds);
            }

            public void Reset()
            {
                _activeGeneration = 0;
                _hasTimeline = false;
            }
        }
        
        private void ApplyBlendShapes()
        {
            if (_resolvedTargets == null || _resolvedTargets.Count == 0)
                return;
            
            // Smooth viseme weights
            for (int i = 0; i < _targetWeights.Length; i++)
            {
                _currentWeights[i] = Mathf.SmoothDamp(
                    _currentWeights[i],
                    _targetWeights[i],
                    ref _velocities[i],
                    _smoothTime
                );
            }
            
            // Apply to all resolved blend shape targets
            for (int i = 0; i < _resolvedTargets.Count; i++)
            {
                var t = _resolvedTargets[i];
                float weight = _currentWeights[t.VisemeIndex] * t.TargetWeight * 100f; // Unity blend shapes are 0-100
                t.Renderer.SetBlendShapeWeight(t.BlendShapeIndex, weight);
            }
        }
        
#if UNITY_EDITOR
        private void OnValidate()
        {
            _playDelay = Mathf.Max(0f, _playDelay);
            _realtimeWindowSeconds = Mathf.Max(0.15f, _realtimeWindowSeconds);
            _realtimeAnalysisInterval = Mathf.Clamp(_realtimeAnalysisInterval, 0.02f, 0.25f);
            _realtimeLookbackSeconds = Mathf.Clamp(_realtimeLookbackSeconds, 0f, 0.12f);
            _realtimeMinRms = Mathf.Clamp(_realtimeMinRms, 0f, 0.1f);
            if (_profile != null && (_targetRoot != null || gameObject.activeInHierarchy))
            {
                Refresh();
            }
        }
#endif
    }
}
