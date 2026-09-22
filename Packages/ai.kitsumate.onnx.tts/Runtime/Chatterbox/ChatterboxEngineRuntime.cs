using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
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
        private string _displayName;

        // Language preprocessor
        private LanguagePreprocessor _languagePreprocessor;

        // Cached voice reference data (set in OnPrepareInput on main thread)
        private float[] _cachedVoiceSamples;
        private EntityId _cachedVoiceClipId;

        // Voice encoder cache (LRU)
        private VoiceEncoderCache _voiceCache;

        private bool _usesModernInputs;
        private bool _isV3;
        private string[] _kvPastNames;
        private Dictionary<string, string> _kvOutputToPastName;
        private string _logitsOutputName;
        private int _logitsOutputIndex;

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
            _displayName = _modelSet.DisplayName;
            _languagePreprocessor = new LanguagePreprocessor(
                _modelSet.CangjieMapping?.text,
                _modelSet.JapaneseReadings?.text,
                _modelSet.RussianStress?.text,
                _modelSet.ChineseWords?.text);
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
                // Large graphs already load on a background thread. Load them one
                // at a time to limit peak memory and retain handles for cleanup.
                var options = new OnnxSessionOptions
                    { IntraOpThreads = Math.Min(4, Environment.ProcessorCount), InterOpThreads = 1 };
                speechEncoderSession = backend.CreateSession(_modelSet.SpeechEncoder, options);
                embedTokensSession = backend.CreateSession(_modelSet.EmbedTokens, options);
                languageModelSession = backend.CreateSession(_modelSet.LanguageModel, options);
                conditionalDecoderSession = backend.CreateSession(_modelSet.ConditionalDecoder, options);
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

            _usesModernInputs = !_embedTokensSession.InputNames.Contains("position_ids");
            _isV3 = _embedTokensSession.InputNames.Contains("text_conditioning");
            _kvPastNames = _languageModelSession.InputNames
                .Where(name => name.StartsWith("past_key_values.", StringComparison.Ordinal))
                .ToArray();
            _kvOutputToPastName = _languageModelSession.OutputNames
                .Where(name => name.StartsWith("present.", StringComparison.Ordinal))
                .ToDictionary(name => name, name => "past_key_values." + name.Substring("present.".Length),
                    StringComparer.Ordinal);
            _logitsOutputName = _languageModelSession.OutputNames.Contains("logits")
                ? "logits"
                : _languageModelSession.OutputNames[0];
            _logitsOutputIndex = _languageModelSession.OutputNames.ToList().IndexOf(_logitsOutputName);

            if (VerboseLogging)
                Debug.Log($"[ChatterboxEngine] Loaded {_displayName} (multilingual={IsMultilingual}, voiceCache={_voiceCacheCapacity})");
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
            _displayName = null;
            _languagePreprocessor = null;
            _voiceCache?.Dispose();
            _voiceCache = null;
            _usesModernInputs = false;
            _isV3 = false;
            _kvPastNames = null;
            _kvOutputToPastName = null;
            _logitsOutputName = null;
            _logitsOutputIndex = 0;
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

            _cachedVoiceSamples = PrepareVoiceSamples(clip);
        }

        protected override TtsResult OnRun(TtsRequest input, System.Threading.CancellationToken cancellationToken)
        {
            var totalSw = VerboseLogging ? Stopwatch.StartNew() : null;
            var phaseSw = VerboseLogging ? Stopwatch.StartNew() : null;

            // 1. Prepare text with language preprocessing
            string text = input.Text;
            string langId = input.LanguageId;
            var generation = input.Chatterbox ?? (_isV3 ? _modelSet.Generation : null);
            generation?.Validate();
            if (!_isV3 && generation?.Guidance > 0)
                throw new ArgumentException("Guidance requires a V3 export with text_conditioning inputs.");
            int batch = generation?.Guidance > 0 ? 2 : 1;
            var random = new System.Random(generation?.Seed ?? 42);
            if (_isV3)
            {
                langId = string.IsNullOrWhiteSpace(langId) ? "en" : langId.ToLowerInvariant();
                if (!ChatterboxConstants.SupportedLanguages.ContainsKey(langId))
                    throw new ArgumentException($"Unsupported Chatterbox language: {langId}");
                text = PrepareV3Text(text);
            }
            if (IsMultilingual && !string.IsNullOrEmpty(langId))
            {
                text = _isV3 ? _languagePreprocessor.ProcessV3(text, langId)
                    : _languagePreprocessor.Process(text, langId);
                text = $"[{langId.ToLowerInvariant()}]{text}";
            }
            else if (_usesModernInputs)
            {
                text = PrepareModernText(text);
            }

            // 2. Tokenize (includes TemplateProcessing framing:
            //    [EXAGGERATION][START] {text} [STOP][START_SPEECH][START_SPEECH])
            if (_isV3) text = text.Replace(" ", "[SPACE]");
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
                    CondEmb = GetOutput(speechEncoderOutputs, "audio_features", 0),
                    PromptToken = GetOutput(speechEncoderOutputs, "audio_tokens", 1),
                    RefXVector = GetOutput(speechEncoderOutputs, "speaker_embeddings", 2),
                    PromptFeat = GetOutput(speechEncoderOutputs, "speaker_features", 3),
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

            int kvCount = _kvPastNames.Length;
            int kvHeads = Math.Max(1, voiceEntry.CondEmb.Shape[2] / ChatterboxConstants.HeadDim);

            OnnxTensor exaggerationTensor = _embedTokensSession.InputNames.Contains("exaggeration")
                ? OnnxTensor.FromArray(new[] { exaggeration }, new[] { 1 })
                : null;
            OnnxTensor speechPositionTensor = _embedTokensSession.InputNames.Contains("position_ids")
                ? OnnxTensor.FromArray(new[] { 0L }, new[] { 1, 1 })
                : null;
            var embedInputs = new Dictionary<string, OnnxTensor>
            {
                ["input_ids"] = null
            };
            if (speechPositionTensor != null) embedInputs["position_ids"] = null;
            if (exaggerationTensor != null) embedInputs["exaggeration"] = exaggerationTensor;

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
                cpuOutputNames = new HashSet<string> { _logitsOutputName };
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
                    kvCache[_kvPastNames[i]] = OnnxTensor.FromArray(
                        Array.Empty<float>(),
                        new[] { batch, kvHeads, 0, ChatterboxConstants.HeadDim });
                }
                lmInputs = new Dictionary<string, OnnxTensor>(3 + kvCount)
                {
                    ["inputs_embeds"] = null,
                    ["attention_mask"] = null
                };
                if (_languageModelSession.InputNames.Contains("position_ids"))
                    lmInputs["position_ids"] = null;
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
                    if (speechPositionTensor != null)
                        embedInputs["position_ids"] = OnnxTensor.FromArray(positionIds, new[] { 1, positionIds.Length });
                }
                else
                {
                    embedInputs["input_ids"] = OnnxTensor.FromArray(
                        new[] { (long)generatedTokens[generatedTokens.Count - 1] }, new[] { 1, 1 });
                    if (speechPositionTensor != null)
                    {
                        speechPositionTensor.Dispose();
                        speechPositionTensor = OnnxTensor.FromArray(new[] { (long)step }, new[] { 1, 1 });
                        embedInputs["position_ids"] = speechPositionTensor;
                    }
                }

                if (_isV3)
                {
                    embedInputs["text_conditioning"] = OnnxTensor.FromArray(
                        batch == 2 ? new[] { 1f, 0f } : new[] { 1f }, new[] { batch });
                    if (batch == 2)
                    {
                        foreach (string key in new[] { "input_ids", "position_ids" })
                        {
                            var data = embedInputs[key].AsLongArray();
                            embedInputs[key] = OnnxTensor.FromArray(data.Concat(data).ToArray(), new[] { batch, data.Length });
                        }
                        embedInputs["exaggeration"] = OnnxTensor.FromArray(new[] { exaggeration, exaggeration }, new[] { batch });
                    }
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
                OnnxTensor lmPositionTensor = null;
                if (_languageModelSession.InputNames.Contains("position_ids"))
                {
                    long[] values = step == 0
                        ? Enumerable.Range(0, initialSeqLen).Select(index => (long)index).ToArray()
                        : new[] { (long)attentionMaskLen - 1 };
                    lmPositionTensor = OnnxTensor.FromArray(
                        batch == 2 ? values.Concat(values).ToArray() : values,
                        new[] { batch, values.Length });
                }

                if (deviceSession != null)
                {
                    // ── IO Binding path: KV cache stays on GPU ──
                    var cpuInputs = new List<OnnxNamedValue>(2)
                    {
                        new OnnxNamedValue("inputs_embeds", finalEmbeds),
                        new OnnxNamedValue("attention_mask", OnnxTensor.FromArray(
                            Enumerable.Repeat(1L, batch * attentionMaskLen).ToArray(),
                            new[] { batch, attentionMaskLen }))
                    };
                    if (lmPositionTensor != null)
                        cpuInputs.Add(new OnnxNamedValue("position_ids", lmPositionTensor));

                    IReadOnlyList<IDeviceTensor> deviceInputs;
                    if (step == 0)
                    {
                        // First step: pass empty KV cache as CPU inputs
                        for (int i = 0; i < kvCount; i++)
                        {
                            cpuInputs.Add(new OnnxNamedValue(_kvPastNames[i],
                                OnnxTensor.FromArray(Array.Empty<float>(),
                                    new[] { batch, kvHeads, 0, ChatterboxConstants.HeadDim })));
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
                    var logitsTensor = lmOutputs[_logitsOutputIndex].ToCpu();
                    logits = logitsTensor.AsFloatArray();
                    vocabSize = logitsTensor.Shape[^1];

                    // Dispose old KV device tensors, store new ones
                    var outputNames = _languageModelSession.OutputNames;
                    for (int i = 0; i < outputNames.Count; i++)
                    {
                        if (i == _logitsOutputIndex) continue;
                        if (!_kvOutputToPastName.TryGetValue(outputNames[i], out string pastName)) continue;
                        int kvIdx = Array.IndexOf(_kvPastNames, pastName);
                        if (kvIdx < 0) continue;
                        kvDeviceTensors[kvIdx]?.Dispose();
                        lmOutputs[i].Name = pastName; // set correct input name for next step
                        kvDeviceTensors[kvIdx] = lmOutputs[i];
                    }

                    // Dispose the logits device tensor (we already extracted the data)
                    lmOutputs[_logitsOutputIndex].Dispose();
                }
                else
                {
                    // ── CPU fallback path (original) ──
                    lmInputs["inputs_embeds"] = finalEmbeds;
                    lmInputs["attention_mask"] = OnnxTensor.FromArray(
                        Enumerable.Repeat(1L, batch * attentionMaskLen).ToArray(),
                        new[] { batch, attentionMaskLen });
                    if (lmPositionTensor != null)
                        lmInputs["position_ids"] = lmPositionTensor;
                    foreach (var kv in kvCache)
                        lmInputs[kv.Key] = kv.Value;

                    var lmOutputs = _languageModelSession.Run(lmInputs);

                    var logitsTensor = lmOutputs[_logitsOutputName];
                    logits = logitsTensor.AsFloatArray();
                    vocabSize = logitsTensor.Shape[^1];

                    var outputNames = _languageModelSession.OutputNames;
                    for (int i = 0; i < outputNames.Count; i++)
                    {
                        if (i == _logitsOutputIndex) continue;
                        if (!_kvOutputToPastName.TryGetValue(outputNames[i], out string pastName)) continue;
                        kvCache[pastName]?.Dispose();
                        kvCache[pastName] = lmOutputs[outputNames[i]];
                    }
                }
                lmPositionTensor?.Dispose();

                if (VerboseLogging)
                {
                    long stepMs = phaseSw.ElapsedMilliseconds;
                    lmTotalMs += stepMs;
                    if (step == 0) lmStep0Ms = stepMs;
                }

                // 5d. Extract last-step logits and apply repetition penalty
                var lastLogits = ChatterboxSampling.Combine(logits, vocabSize, batch, generation?.Guidance ?? 0);

                repetitionPenalty.Apply(generatedTokens, lastLogits);

                int nextToken = ChatterboxSampling.Sample(lastLogits, generation, random);

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
            if (_isV3)
            {
                // Upstream drops non-codec tokens before S3Gen, including out-of-band vocabulary entries.
                bool stopped = generatedTokens[^1] == ChatterboxConstants.StopSpeechToken;
                generatedTokens = generatedTokens.Take(1).Concat(generatedTokens.Skip(1)
                    .Where(token => token >= 0 && token < ChatterboxConstants.StartSpeechToken)).ToList();
                if (stopped) generatedTokens.Add(ChatterboxConstants.StopSpeechToken);
            }
            var promptTokenData = voiceEntry.PromptToken.AsLongArray();
            int genStart = 1; // skip START_SPEECH_TOKEN
            int genEnd = generatedTokens.Count;
            if (generatedTokens[genEnd - 1] == ChatterboxConstants.StopSpeechToken)
                genEnd--;
            if (genEnd == genStart)
                throw new InvalidOperationException("Chatterbox generated no speech tokens.");

            int silenceCount = _usesModernInputs ? 3 : 0;
            int speechCount = promptTokenData.Length + (genEnd - genStart) + silenceCount;
            var speechTokenIds = new long[speechCount];
            Array.Copy(promptTokenData, 0, speechTokenIds, 0, promptTokenData.Length);
            for (int i = genStart; i < genEnd; i++)
                speechTokenIds[promptTokenData.Length + i - genStart] = generatedTokens[i];
            for (int i = speechCount - silenceCount; i < speechCount; i++)
                speechTokenIds[i] = ChatterboxConstants.SilenceToken;

            // 7. Run conditional decoder
            var decoderInputs = new Dictionary<string, OnnxTensor>
            {
                ["speech_tokens"] = OnnxTensor.FromArray(speechTokenIds, new[] { 1, speechCount }),
                ["speaker_embeddings"] = voiceEntry.RefXVector,
                ["speaker_features"] = voiceEntry.PromptFeat
            };

            var decoderOutputs = _conditionalDecoderSession.Run(decoderInputs);
            var wavData = GetOutput(decoderOutputs, "waveform", 0).AsFloatArray();

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

        internal static string PrepareV3Text(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Chatterbox requires nonempty text.");
            text = string.Join(" ", text.Split((char[])null, StringSplitOptions.RemoveEmptyEntries));
            foreach (var pair in new[] { ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "),
                (";", ", "), ("—", "-"), ("–", "-"), (" ,", ","), ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'") })
                text = text.Replace(pair.Item1, pair.Item2);
            text = text.TrimEnd();
            if (!".!?-,、，。？！".Contains(text[^1])) text += ".";
            return text;
        }

        private static string PrepareModernText(string text)
        {
            if (string.IsNullOrEmpty(text)) return "You need to add some text for me to talk.";
            text = string.Join(" ", text.Split((char[])null, StringSplitOptions.RemoveEmptyEntries));
            foreach (var replacement in new[]
            {
                ("…", ", "), (":", ","), ("—", "-"), ("–", "-"), (" ,", ","),
                ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'")
            })
                text = text.Replace(replacement.Item1, replacement.Item2);
            text = text.TrimEnd();
            if (char.IsLower(text[0])) text = char.ToUpperInvariant(text[0]) + text.Substring(1);
            if (!".!?-,".Contains(text[text.Length - 1])) text += ".";
            return text;
        }

        private static OnnxTensor GetOutput(IReadOnlyDictionary<string, OnnxTensor> outputs, string name,
            int fallbackIndex)
        {
            if (outputs.TryGetValue(name, out OnnxTensor output)) return output;
            return outputs.ElementAt(fallbackIndex).Value;
        }

        /// <summary>
        /// Concatenates two 3D tensors along axis 1 (sequence dimension).
        /// Shapes: [batch, seq1, hidden] + [batch, seq2, hidden] → [batch, seq1+seq2, hidden].
        /// </summary>
        private static OnnxTensor ConcatEmbeddings(OnnxTensor a, OnnxTensor b)
        {
            var aData = a.AsFloatArray();
            var bData = b.AsFloatArray();

            int batch = b.Shape[0];
            int seq1 = a.Shape[1];
            int seq2 = b.Shape[1];
            int hidden = a.Shape[2];

            var result = new float[batch * (seq1 + seq2) * hidden];
            for (int ba = 0; ba < batch; ba++)
            {
                Array.Copy(aData, (a.Shape[0] == 1 ? 0 : ba) * seq1 * hidden, result, ba * (seq1 + seq2) * hidden, seq1 * hidden);
                Array.Copy(bData, ba * seq2 * hidden, result, ba * (seq1 + seq2) * hidden + seq1 * hidden, seq2 * hidden);
            }

            return OnnxTensor.FromArray(result, new[] { batch, seq1 + seq2, hidden });
        }

        /// <summary>Reads, downmixes, and resamples a reference clip for Chatterbox encoders.</summary>
        internal static float[] PrepareVoiceSamples(AudioClip clip)
        {
            if (clip == null) throw new ArgumentNullException(nameof(clip));
            if (clip.samples <= 0 || clip.channels <= 0 || clip.frequency <= 0)
                throw new ArgumentException("The reference voice has no usable audio samples.", nameof(clip));
            var interleaved = new float[checked(clip.samples * clip.channels)];
            if (!clip.GetData(interleaved, 0))
                throw new InvalidOperationException("Could not read reference voice samples.");
            var mono = new float[clip.samples];
            for (int sample = 0; sample < mono.Length; sample++)
                for (int channel = 0; channel < clip.channels; channel++)
                    mono[sample] += interleaved[sample * clip.channels + channel] / clip.channels;
            return Resample(mono, clip.frequency, ChatterboxConstants.SampleRate);
        }

        /// <summary>Linear interpolation resampling.</summary>
        private static float[] Resample(float[] samples, int srcRate, int dstRate)
        {
            if (srcRate == dstRate) return samples;

            double ratio = (double)srcRate / dstRate;
            int newLength = (int)(samples.Length / ratio);
            if (newLength == 0)
                throw new ArgumentException("The reference voice is too short after resampling.", nameof(samples));
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
