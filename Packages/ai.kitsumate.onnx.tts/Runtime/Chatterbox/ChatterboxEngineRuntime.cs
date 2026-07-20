using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using KitsuMate.Tokenizers;
using UnityEngine;
using KitsuMate.Onnx;
using Debug = UnityEngine.Debug;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    /// <summary>
    /// Chatterbox TTS engine using 4 ONNX models in an autoregressive pipeline:
    /// speech_encoder → embed_tokens → language_model (Llama) → conditional_decoder.
    /// Supports both English-only and multilingual variants.
    /// </summary>
    public sealed class ChatterboxEngineRuntime : TtsEngineRuntime
    {
        [Header("Model Set")]
        private readonly ChatterboxModelSet _modelSet;

        [Header("Performance")]
        [SerializeField, Tooltip("Number of voice embeddings to cache (0 = disabled)")]
        private int _voiceCacheCapacity = 4;

        /// <summary>Model set containing Chatterbox models.</summary>
        public ChatterboxEngineRuntime(ChatterboxModelSet modelSet, int voiceCacheCapacity, bool verbose)
        {
            _modelSet = modelSet;
            _voiceCacheCapacity = voiceCacheCapacity;
            VerboseLogging = verbose;
        }

        // ONNX sessions
        private IOnnxSession _speechEncoderSession;
        private IOnnxSession _embedTokensSession;
        private IOnnxSession _languageModelSession;
        private IOnnxSession _conditionalDecoderSession;

        // Tokenizer (initialized on load)
        private Tokenizer _tokenizer;
        private bool _isMultilingual;

        // Language preprocessor
        private LanguagePreprocessor _languagePreprocessor;

        // Cached voice reference data (set in OnPrepareInput on main thread)
        private float[] _cachedVoiceSamples;
        private EntityId _cachedVoiceClipId;

        // Voice encoder cache (LRU)
        private VoiceEncoderCache _voiceCache;

        // Pre-computed KV cache output name → past_key_values name mapping
        private string[] _kvOutputToPastName;

        public override int OutputSampleRate => ChatterboxConstants.SampleRate;
        public override bool IsMultilingual => _isMultilingual;

        public override string[] SupportedLanguages =>
            IsMultilingual
                ? new List<string>(ChatterboxConstants.SupportedLanguages.Keys).ToArray()
                : Array.Empty<string>();

        protected override void OnLoadMainThread(OnnxBackend backend)
        {
            if (_modelSet == null || !_modelSet.IsComplete)
                throw new InvalidOperationException("ChatterboxModelSet is not assigned or incomplete");

            _tokenizer = Tokenizer.FromTokenizerJson(_modelSet.Tokenizer.bytes);
            _isMultilingual = DetectMultilingual(_modelSet.Tokenizer.text);
            _languagePreprocessor = new LanguagePreprocessor(
                _modelSet.CangjieMapping != null ? _modelSet.CangjieMapping.text : null);
        }

        protected override void OnLoadBackground(OnnxBackend backend)
        {
            if (_modelSet == null || !_modelSet.IsComplete)
                throw new InvalidOperationException("ChatterboxModelSet is not assigned or incomplete");

            IOnnxSession speechEncoderSession = null;
            IOnnxSession embedTokensSession = null;
            IOnnxSession languageModelSession = null;
            IOnnxSession conditionalDecoderSession = null;

            try
            {
                var speechEncoderTask = Task.Run(() => backend.CreateSession(_modelSet.SpeechEncoder));
                var embedTokensTask = Task.Run(() => backend.CreateSession(_modelSet.EmbedTokens));
                var languageModelTask = Task.Run(() => backend.CreateSession(_modelSet.LanguageModel));
                var conditionalDecoderTask = Task.Run(() => backend.CreateSession(_modelSet.ConditionalDecoder));

                Task.WhenAll(speechEncoderTask, embedTokensTask, languageModelTask, conditionalDecoderTask)
                    .GetAwaiter()
                    .GetResult();

                speechEncoderSession = speechEncoderTask.Result;
                embedTokensSession = embedTokensTask.Result;
                languageModelSession = languageModelTask.Result;
                conditionalDecoderSession = conditionalDecoderTask.Result;
            }
            catch
            {
                speechEncoderSession?.Dispose();
                embedTokensSession?.Dispose();
                languageModelSession?.Dispose();
                conditionalDecoderSession?.Dispose();
                throw;
            }

            _speechEncoderSession = speechEncoderSession;
            _embedTokensSession = embedTokensSession;
            _languageModelSession = languageModelSession;
            _conditionalDecoderSession = conditionalDecoderSession;

            // Propagate verbose logging to sessions
            if (VerboseLogging)
            {
                _speechEncoderSession.VerboseLogging = true;
                _embedTokensSession.VerboseLogging = true;
                _languageModelSession.VerboseLogging = true;
                _conditionalDecoderSession.VerboseLogging = true;
            }

            // Initialize voice encoder cache
            _voiceCache = _voiceCacheCapacity > 0 ? new VoiceEncoderCache(_voiceCacheCapacity) : null;

            // Pre-compute KV output name mapping: "present.X.Y" → "past_key_values.X.Y"
            var outputNames = _languageModelSession.OutputNames;
            _kvOutputToPastName = new string[outputNames.Count];
            for (int i = 1; i < outputNames.Count; i++)
                _kvOutputToPastName[i] = outputNames[i].Replace("present", "past_key_values");

            if (VerboseLogging)
                Debug.Log($"[ChatterboxEngine] Loaded {_modelSet.DisplayName} (multilingual={IsMultilingual}, voiceCache={_voiceCacheCapacity})");
        }

        protected override void OnUnload()
        {
            _speechEncoderSession?.Dispose();
            _embedTokensSession?.Dispose();
            _languageModelSession?.Dispose();
            _conditionalDecoderSession?.Dispose();

            _speechEncoderSession = null;
            _embedTokensSession = null;
            _languageModelSession = null;
            _conditionalDecoderSession = null;
            _tokenizer = null;
            _isMultilingual = false;
            _languagePreprocessor = null;
            _voiceCache?.Dispose();
            _voiceCache = null;
            _kvOutputToPastName = null;
        }

        /// <summary>
        /// Cache AudioClip data on main thread before switching to background.
        /// </summary>
        protected override void OnPrepareInput(TtsRequest input)
        {
            var clip = input.VoiceReference ?? _modelSet.DefaultVoice;
            if (clip == null)
                throw new InvalidOperationException("No voice reference provided and no default voice in model set");

            _cachedVoiceClipId = clip.GetEntityId();

            // Skip sample extraction if voice is already cached
            if (_voiceCache != null && _voiceCache.Contains(_cachedVoiceClipId))
            {
                _cachedVoiceSamples = null;
                return;
            }

            var samples = new float[clip.samples * clip.channels];
            clip.GetData(samples, 0);

            // Convert to mono
            if (clip.channels > 1)
            {
                var mono = new float[clip.samples];
                for (int i = 0; i < clip.samples; i++)
                {
                    float sum = 0f;
                    for (int ch = 0; ch < clip.channels; ch++)
                        sum += samples[i * clip.channels + ch];
                    mono[i] = sum / clip.channels;
                }
                samples = mono;
            }

            // Resample to 24kHz if needed
            if (clip.frequency != ChatterboxConstants.SampleRate)
                samples = Resample(samples, clip.frequency, ChatterboxConstants.SampleRate);

            _cachedVoiceSamples = samples;
        }

        protected override TtsResult OnRun(TtsRequest input, System.Threading.CancellationToken cancellationToken)
        {
            var totalSw = VerboseLogging ? Stopwatch.StartNew() : null;
            var phaseSw = VerboseLogging ? Stopwatch.StartNew() : null;

            // 1. Prepare text with language preprocessing
            string text = input.Text;
            string langId = input.LanguageId;
            if (IsMultilingual && !string.IsNullOrEmpty(langId))
            {
                text = _languagePreprocessor.Process(text, langId);
                text = $"[{langId.ToLowerInvariant()}]{text}";
            }

            // 2. Tokenize (includes TemplateProcessing framing:
            //    [EXAGGERATION][START] {text} [STOP][START_SPEECH][START_SPEECH])
            var encoded = _tokenizer.Encode(text, addSpecialTokens: true);
            long[] inputIds = encoded.Ids.Select(static id => (long)id).ToArray();

            // 3. Compute position_ids: sequential for text tokens, reset to 0 for speech tokens
            long[] positionIds = new long[inputIds.Length];
            for (int i = 0; i < inputIds.Length; i++)
            {
                positionIds[i] = inputIds[i] >= ChatterboxConstants.StartSpeechToken
                    ? 0
                    : Math.Max(0, i - 1);
            }

            if (VerboseLogging)
            {
                Debug.Log($"[ChatterboxEngine] Tokenize: {phaseSw.ElapsedMilliseconds}ms ({inputIds.Length} tokens)");
                phaseSw.Restart();
            }

            // 4. Get speech encoder outputs (cached or computed)
            VoiceCacheEntry voiceEntry;
            if (_voiceCache != null && _voiceCache.TryGet(_cachedVoiceClipId, out voiceEntry))
            {
                if (VerboseLogging)
                    Debug.Log($"[ChatterboxEngine] Voice cache hit (clipId={_cachedVoiceClipId})");
            }
            else
            {
                var speechEncoderInputs = new Dictionary<string, OnnxTensor>
                {
                    ["audio_values"] = OnnxTensor.FromArray(_cachedVoiceSamples, new[] { 1, _cachedVoiceSamples.Length })
                };

                var speechEncoderOutputs = _speechEncoderSession.Run(speechEncoderInputs);
                voiceEntry = new VoiceCacheEntry
                {
                    CondEmb = speechEncoderOutputs[_speechEncoderSession.OutputNames[0]],
                    PromptToken = speechEncoderOutputs[_speechEncoderSession.OutputNames[1]],
                    RefXVector = speechEncoderOutputs[_speechEncoderSession.OutputNames[2]],
                    PromptFeat = speechEncoderOutputs[_speechEncoderSession.OutputNames[3]],
                };
                _voiceCache?.Put(_cachedVoiceClipId, voiceEntry);

                if (VerboseLogging)
                    Debug.Log($"[ChatterboxEngine] Voice cache miss → encoded (clipId={_cachedVoiceClipId})");
            }

            if (VerboseLogging)
            {
                Debug.Log($"[ChatterboxEngine] Speech encoder: {phaseSw.ElapsedMilliseconds}ms");
                phaseSw.Restart();
            }

            // 5. Autoregressive generation loop
            float exaggeration = input.Exaggeration;
            int maxNewTokens = input.MaxNewTokens > 0 ? input.MaxNewTokens : ChatterboxConstants.DefaultMaxNewTokens;
            var repetitionPenalty = new RepetitionPenaltyProcessor(
                input.RepetitionPenalty > 0 ? input.RepetitionPenalty : ChatterboxConstants.DefaultRepetitionPenalty);

            var generatedTokens = new List<int>(maxNewTokens + 1) { ChatterboxConstants.StartSpeechToken };

            // Initialize KV cache — 30 layers × 2 (key, value), shape [1, 16, 0, 64]
            int kvCount = ChatterboxConstants.NumHiddenLayers * 2;
            var kvPastNames = new string[kvCount];
            {
                int idx = 0;
                for (int layer = 0; layer < ChatterboxConstants.NumHiddenLayers; layer++)
                {
                    kvPastNames[idx++] = $"past_key_values.{layer}.key";
                    kvPastNames[idx++] = $"past_key_values.{layer}.value";
                }
            }

            // Pre-allocate reusable tensors and dictionaries
            var exaggerationTensor = OnnxTensor.FromArray(new[] { exaggeration }, new[] { 1 });
            var speechPositionTensor = OnnxTensor.FromArray(new[] { 0L }, new[] { 1, 1 });

            var embedInputs = new Dictionary<string, OnnxTensor>(3)
            {
                ["input_ids"] = null,
                ["position_ids"] = null,
                ["exaggeration"] = exaggerationTensor
            };

            // Pre-allocate attention mask as long[] — max possible length
            int initialSeqLen = 0; // set after step 0
            long[] attentionMask = null;
            int attentionMaskLen = 0;

            long embedTotalMs = 0, lmTotalMs = 0, lmStep0Ms = 0;

            // Use IO Binding path when the session supports device tensors (GPU passthrough)
            var deviceSession = _languageModelSession as IOnnxDeviceSession;

            // Build the set of output names that should stay on CPU (only logits)
            // All other outputs (KV cache) stay on device.
            HashSet<string> cpuOutputNames = null;
            if (deviceSession != null)
            {
                cpuOutputNames = new HashSet<string> { _languageModelSession.OutputNames[0] };
            }

            // Device tensor KV cache (IO Binding path)
            IDeviceTensor[] kvDeviceTensors = deviceSession != null ? new IDeviceTensor[kvCount] : null;

            // CPU tensor KV cache (fallback path)
            Dictionary<string, OnnxTensor> kvCache = null;
            Dictionary<string, OnnxTensor> lmInputs = null;
            if (deviceSession == null)
            {
                kvCache = new Dictionary<string, OnnxTensor>(kvCount);
                for (int i = 0; i < kvCount; i++)
                {
                    kvCache[kvPastNames[i]] = OnnxTensor.FromArray(
                        Array.Empty<float>(),
                        new[] { 1, ChatterboxConstants.NumKeyValueHeads, 0, ChatterboxConstants.HeadDim });
                }
                lmInputs = new Dictionary<string, OnnxTensor>(2 + kvCount)
                {
                    ["inputs_embeds"] = null,
                    ["attention_mask"] = null
                };
                foreach (var kv in kvCache)
                    lmInputs[kv.Key] = kv.Value;
            }

            try
            {
            for (int step = 0; step < maxNewTokens; step++)
            {
                ThrowIfCancellationRequested();
                if (VerboseLogging) phaseSw.Restart();

                // 5a. Run embed_tokens
                if (step == 0)
                {
                    embedInputs["input_ids"] = OnnxTensor.FromArray(inputIds, new[] { 1, inputIds.Length });
                    embedInputs["position_ids"] = OnnxTensor.FromArray(positionIds, new[] { 1, positionIds.Length });
                }
                else
                {
                    embedInputs["input_ids"] = OnnxTensor.FromArray(
                        new[] { (long)generatedTokens[generatedTokens.Count - 1] }, new[] { 1, 1 });
                    embedInputs["position_ids"] = speechPositionTensor;
                }

                var embedOutputs = _embedTokensSession.Run(embedInputs);
                var inputsEmbeds = embedOutputs[_embedTokensSession.OutputNames[0]];

                if (VerboseLogging) embedTotalMs += phaseSw.ElapsedMilliseconds;

                // 5b. First step: concatenate cond_emb + inputs_embeds
                OnnxTensor finalEmbeds;
                if (step == 0)
                {
                    finalEmbeds = ConcatEmbeddings(voiceEntry.CondEmb, inputsEmbeds);
                    initialSeqLen = finalEmbeds.Shape[1];
                    attentionMask = new long[initialSeqLen + maxNewTokens];
                    for (int i = 0; i < initialSeqLen; i++)
                        attentionMask[i] = 1L;
                    attentionMaskLen = initialSeqLen;
                }
                else
                {
                    finalEmbeds = inputsEmbeds;
                    attentionMask[attentionMaskLen] = 1L;
                    attentionMaskLen++;
                }

                if (VerboseLogging) phaseSw.Restart();

                float[] logits;
                int vocabSize;

                if (deviceSession != null)
                {
                    // ── IO Binding path: KV cache stays on GPU ──
                    var cpuInputs = new List<OnnxNamedValue>(2)
                    {
                        new OnnxNamedValue("inputs_embeds", finalEmbeds),
                        new OnnxNamedValue("attention_mask", OnnxTensor.FromArray(
                            new ReadOnlySpan<long>(attentionMask, 0, attentionMaskLen).ToArray(),
                            new[] { 1, attentionMaskLen }))
                    };

                    IReadOnlyList<IDeviceTensor> deviceInputs;
                    if (step == 0)
                    {
                        // First step: pass empty KV cache as CPU inputs
                        for (int i = 0; i < kvCount; i++)
                        {
                            cpuInputs.Add(new OnnxNamedValue(kvPastNames[i],
                                OnnxTensor.FromArray(Array.Empty<float>(),
                                    new[] { 1, ChatterboxConstants.NumKeyValueHeads, 0, ChatterboxConstants.HeadDim })));
                        }
                        deviceInputs = Array.Empty<IDeviceTensor>();
                    }
                    else
                    {
                        // Subsequent steps: pass KV cache as device tensors
                        deviceInputs = kvDeviceTensors;
                    }

                    var lmOutputs = deviceSession.RunOnDevice(cpuInputs, deviceInputs, cpuOutputNames);

                    // First output is logits (on CPU), remaining are KV cache (on device)
                    var logitsTensor = lmOutputs[0].ToCpu();
                    logits = logitsTensor.AsFloatArray();
                    vocabSize = logitsTensor.Shape[^1];

                    // Dispose old KV device tensors, store new ones
                    var outputNames = _languageModelSession.OutputNames;
                    for (int i = 1; i < outputNames.Count; i++)
                    {
                        // Map output index to KV past name index
                        string pastName = _kvOutputToPastName[i];
                        int kvIdx = Array.IndexOf(kvPastNames, pastName);
                        kvDeviceTensors[kvIdx]?.Dispose();
                        lmOutputs[i].Name = pastName; // set correct input name for next step
                        kvDeviceTensors[kvIdx] = lmOutputs[i];
                    }

                    // Dispose the logits device tensor (we already extracted the data)
                    lmOutputs[0].Dispose();
                }
                else
                {
                    // ── CPU fallback path (original) ──
                    lmInputs["inputs_embeds"] = finalEmbeds;
                    lmInputs["attention_mask"] = OnnxTensor.FromArray(
                        new ReadOnlySpan<long>(attentionMask, 0, attentionMaskLen).ToArray(),
                        new[] { 1, attentionMaskLen });
                    foreach (var kv in kvCache)
                        lmInputs[kv.Key] = kv.Value;

                    var lmOutputs = _languageModelSession.Run(lmInputs);

                    var logitsTensor = lmOutputs[_languageModelSession.OutputNames[0]];
                    logits = logitsTensor.AsFloatArray();
                    vocabSize = logitsTensor.Shape[^1];

                    var outputNames = _languageModelSession.OutputNames;
                    for (int i = 1; i < outputNames.Count; i++)
                        kvCache[_kvOutputToPastName[i]] = lmOutputs[outputNames[i]];
                }

                if (VerboseLogging)
                {
                    long stepMs = phaseSw.ElapsedMilliseconds;
                    lmTotalMs += stepMs;
                    if (step == 0) lmStep0Ms = stepMs;
                }

                // 5d. Extract last-step logits and apply repetition penalty
                int logitsOffset = (logits.Length / vocabSize - 1) * vocabSize;
                var lastLogits = new float[vocabSize];
                Array.Copy(logits, logitsOffset, lastLogits, 0, vocabSize);

                repetitionPenalty.Apply(generatedTokens, lastLogits);

                // 5e. Argmax
                int nextToken = 0;
                float maxVal = float.NegativeInfinity;
                for (int i = 0; i < lastLogits.Length; i++)
                {
                    if (lastLogits[i] > maxVal)
                    {
                        maxVal = lastLogits[i];
                        nextToken = i;
                    }
                }

                generatedTokens.Add(nextToken);

                if (nextToken == ChatterboxConstants.StopSpeechToken)
                    break;
            }

            } // end try
            finally
            {
                // Dispose all local device/tensor resources — handles normal completion,
                // cancellation during domain reload, and unexpected exceptions.
                exaggerationTensor?.Dispose();
                speechPositionTensor?.Dispose();

                if (kvDeviceTensors != null)
                {
                    for (int i = 0; i < kvDeviceTensors.Length; i++)
                    {
                        kvDeviceTensors[i]?.Dispose();
                        kvDeviceTensors[i] = null;
                    }
                }

                if (kvCache != null)
                {
                    foreach (var kv in kvCache)
                        kv.Value?.Dispose();
                    kvCache.Clear();
                }
            }

            int totalSteps = generatedTokens.Count - 1;
            if (VerboseLogging)
            {
                Debug.Log($"[ChatterboxEngine] Generation loop: {totalSteps} steps");
                Debug.Log($"[ChatterboxEngine]   embed_tokens total: {embedTotalMs}ms (avg {(totalSteps > 0 ? embedTotalMs / totalSteps : 0)}ms/step)");
                Debug.Log($"[ChatterboxEngine]   language_model total: {lmTotalMs}ms (step0={lmStep0Ms}ms, avg_step1+={(totalSteps > 1 ? (lmTotalMs - lmStep0Ms) / (totalSteps - 1) : 0)}ms/step)");
                phaseSw.Restart();
            }

            // 6. Build speech tokens: remove START/STOP markers, prepend prompt_token
            var promptTokenData = voiceEntry.PromptToken.AsLongArray();
            int genStart = 1; // skip START_SPEECH_TOKEN
            int genEnd = generatedTokens.Count;
            if (generatedTokens[genEnd - 1] == ChatterboxConstants.StopSpeechToken)
                genEnd--;

            int speechCount = promptTokenData.Length + (genEnd - genStart);
            var speechTokenIds = new long[speechCount];
            Array.Copy(promptTokenData, 0, speechTokenIds, 0, promptTokenData.Length);
            for (int i = genStart; i < genEnd; i++)
                speechTokenIds[promptTokenData.Length + i - genStart] = generatedTokens[i];

            // 7. Run conditional decoder
            var decoderInputs = new Dictionary<string, OnnxTensor>
            {
                ["speech_tokens"] = OnnxTensor.FromArray(speechTokenIds, new[] { 1, speechCount }),
                ["speaker_embeddings"] = voiceEntry.RefXVector,
                ["speaker_features"] = voiceEntry.PromptFeat
            };

            var decoderOutputs = _conditionalDecoderSession.Run(decoderInputs);
            var wavData = decoderOutputs[_conditionalDecoderSession.OutputNames[0]].AsFloatArray();

            if (VerboseLogging)
            {
                Debug.Log($"[ChatterboxEngine] Conditional decoder: {phaseSw.ElapsedMilliseconds}ms ({speechCount} speech tokens → {wavData.Length} samples, {wavData.Length / (float)ChatterboxConstants.SampleRate:F2}s audio)");
                Debug.Log($"[ChatterboxEngine] Total synthesis: {totalSw.ElapsedMilliseconds}ms");
            }

            return new TtsResult
            {
                Samples = wavData,
                SampleRate = ChatterboxConstants.SampleRate,
                TokensGenerated = genEnd - genStart
            };
        }

        private static bool DetectMultilingual(string tokenizerJson)
        {
            return tokenizerJson.Contains("\"[ko]\"") || tokenizerJson.Contains("\"[zh]\"");
        }

        /// <summary>
        /// Concatenates two 3D tensors along axis 1 (sequence dimension).
        /// Shapes: [batch, seq1, hidden] + [batch, seq2, hidden] → [batch, seq1+seq2, hidden].
        /// </summary>
        private static OnnxTensor ConcatEmbeddings(OnnxTensor a, OnnxTensor b)
        {
            var aData = a.AsFloatArray();
            var bData = b.AsFloatArray();

            int batch = a.Shape[0];
            int seq1 = a.Shape[1];
            int seq2 = b.Shape[1];
            int hidden = a.Shape[2];

            var result = new float[batch * (seq1 + seq2) * hidden];
            for (int ba = 0; ba < batch; ba++)
            {
                Array.Copy(aData, ba * seq1 * hidden, result, ba * (seq1 + seq2) * hidden, seq1 * hidden);
                Array.Copy(bData, ba * seq2 * hidden, result, ba * (seq1 + seq2) * hidden + seq1 * hidden, seq2 * hidden);
            }

            return OnnxTensor.FromArray(result, new[] { batch, seq1 + seq2, hidden });
        }

        /// <summary>
        /// Linear interpolation resampling.
        /// </summary>
        private static float[] Resample(float[] samples, int srcRate, int dstRate)
        {
            if (srcRate == dstRate) return samples;

            double ratio = (double)srcRate / dstRate;
            int newLength = (int)(samples.Length / ratio);
            var result = new float[newLength];

            for (int i = 0; i < newLength; i++)
            {
                double srcIndex = i * ratio;
                int idx = (int)srcIndex;
                double frac = srcIndex - idx;

                if (idx + 1 < samples.Length)
                    result[i] = (float)(samples[idx] * (1.0 - frac) + samples[idx + 1] * frac);
                else
                    result[i] = samples[Math.Min(idx, samples.Length - 1)];
            }

            return result;
        }

        #region Voice Encoder Cache

        private struct VoiceCacheEntry
        {
            public OnnxTensor CondEmb;
            public OnnxTensor PromptToken;
            public OnnxTensor RefXVector;
            public OnnxTensor PromptFeat;
        }

        /// <summary>
        /// LRU cache for speech encoder outputs, keyed by AudioClip EntityId.
        /// </summary>
        private class VoiceEncoderCache : IDisposable
        {
            private readonly int _capacity;
            private readonly LinkedList<(EntityId key, VoiceCacheEntry entry)> _list = new();
            private readonly Dictionary<EntityId, LinkedListNode<(EntityId key, VoiceCacheEntry entry)>> _map = new();

            public VoiceEncoderCache(int capacity)
            {
                _capacity = Math.Max(1, capacity);
            }

            public bool Contains(EntityId clipId) => _map.ContainsKey(clipId);

            public bool TryGet(EntityId clipId, out VoiceCacheEntry entry)
            {
                if (_map.TryGetValue(clipId, out var node))
                {
                    // Move to front (most recently used)
                    _list.Remove(node);
                    _list.AddFirst(node);
                    entry = node.Value.entry;
                    return true;
                }
                entry = default;
                return false;
            }

            public void Put(EntityId clipId, VoiceCacheEntry entry)
            {
                if (_map.TryGetValue(clipId, out var existing))
                {
                    DisposeEntry(existing.Value.entry);
                    _list.Remove(existing);
                    _map.Remove(clipId);
                }
                else if (_map.Count >= _capacity)
                {
                    // Evict least recently used
                    var last = _list.Last;
                    DisposeEntry(last.Value.entry);
                    _map.Remove(last.Value.key);
                    _list.RemoveLast();
                }

                var node = _list.AddFirst((clipId, entry));
                _map[clipId] = node;
            }

            public void Dispose()
            {
                foreach (var item in _list)
                    DisposeEntry(item.entry);
                _list.Clear();
                _map.Clear();
            }

            private static void DisposeEntry(VoiceCacheEntry entry)
            {
                entry.CondEmb?.Dispose();
                entry.PromptToken?.Dispose();
                entry.RefXVector?.Dispose();
                entry.PromptFeat?.Dispose();
            }
        }

        #endregion
    }
}
