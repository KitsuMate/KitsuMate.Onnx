using System;
using System.Collections.Generic;
using System.Linq;
using KitsuMate.Tokenizers;
using UnityEngine;
using KitsuMate.Onnx;

namespace KitsuMate.Onnx.Asr.Whisper
{
    /// <summary>
    /// Whisper-based automatic speech recognition engine.
    /// Uses OpenAI Whisper models converted to ONNX format.
    /// </summary>
    public sealed class WhisperEngineRuntime : AsrEngineRuntime
    {
        private readonly IOnnxModelSource _melSource;
        private readonly IOnnxModelSource _encoderSource;
        private readonly IOnnxModelSource _decoderSource;
        private readonly IOnnxModelSource _decoderWithPastSource;
        private readonly TextAsset _tokenizerJson;
        
        [Header("Transcription Settings")]
        [SerializeField, Tooltip("Language code (e.g., 'en', 'ja'). Leave empty for auto-detect.")]
        private string _languageOverride = "";
        
        [SerializeField, Tooltip("Task: transcribe or translate to English")]
        private WhisperTask _task = WhisperTask.Transcribe;
        
        [SerializeField, Tooltip("Maximum number of tokens to generate")]
        private int _maxTokens = 224;
        private AudioClip _preparedClip;
        private float[] _preparedSamples;
        private float _preparedDuration;
        
        // Runtime state
        private IOnnxSession _melSession;
        private IOnnxSession _encoderSession;
        private IOnnxSession _decoderSession;
        private IOnnxSession _decoderWithPastSession;
        private Tokenizer _tokenizer;
        private readonly OnnxSessionOptions _sessionOptions;
        
        public WhisperEngineRuntime(IOnnxModelSource melSource, IOnnxModelSource encoderSource,
            IOnnxModelSource decoderSource, IOnnxModelSource decoderWithPastSource, TextAsset tokenizerJson,
            string languageOverride, WhisperTask task, int maxTokens, bool verbose, OnnxSessionOptions sessionOptions = null)
        {
            _melSource = melSource;
            _encoderSource = encoderSource;
            _decoderSource = decoderSource;
            _decoderWithPastSource = decoderWithPastSource;
            _tokenizerJson = tokenizerJson;
            _languageOverride = languageOverride ?? string.Empty;
            _task = task;
            _maxTokens = maxTokens;
            VerboseLogging = verbose;
            _sessionOptions = sessionOptions;
        }
        
        /// <summary>Language override (empty for auto-detect).</summary>
        public string LanguageOverride
        {
            get => _languageOverride;
            set => _languageOverride = value;
        }
        
        /// <summary>Transcription task.</summary>
        public WhisperTask Task
        {
            get => _task;
            set => _task = value;
        }
        
        public override IReadOnlyList<string> SupportedLanguages => 
            WhisperConstants.SupportedLanguages.ToList();

        protected override void OnLoadMainThread(OnnxBackend backend)
        {
            ValidateSources();
            _tokenizer = Tokenizer.FromTokenizerJson(_tokenizerJson.bytes);
        }
        
        protected override void OnLoadBackground(OnnxBackend backend)
        {
            ValidateSources();
            _melSession = backend.CreateSession(_melSource, _sessionOptions);
            _encoderSession = backend.CreateSession(_encoderSource, _sessionOptions);
            _decoderSession = backend.CreateSession(_decoderSource, _sessionOptions);
            if (_decoderWithPastSource != null && _decoderWithPastSource.IsAvailable)
                _decoderWithPastSession = backend.CreateSession(_decoderWithPastSource, _sessionOptions);
            
            if (VerboseLogging)
                Debug.Log($"[WhisperEngine] Loaded {_encoderSource.SourceName}");
        }
        
        protected override void OnUnload()
        {
            _melSession?.Dispose();
            _encoderSession?.Dispose();
            _decoderSession?.Dispose();
            _decoderWithPastSession?.Dispose();
            
            _melSession = null;
            _encoderSession = null;
            _decoderSession = null;
            _decoderWithPastSession = null;
            _tokenizer = null;
        }

        private void ValidateSources()
        {
            if (_melSource == null || !_melSource.IsAvailable ||
                _encoderSource == null || !_encoderSource.IsAvailable ||
                _decoderSource == null || !_decoderSource.IsAvailable || _tokenizerJson == null)
                throw new InvalidOperationException("Whisper models and tokenizer must be assigned before loading.");
        }
        
        protected override TranscriptionResult Transcribe(AudioClip input)
        {
            // Get audio samples
            var samples = ExtractSamples(input);
            
            // Process mel spectrogram
            var melFeatures = ComputeMelSpectrogram(samples);
            
            // Encode audio
            var encoderOutput = EncodeAudio(melFeatures);
            
            // Decode to text
            var (tokens, detectedLanguage) = DecodeTokens(encoderOutput);
            
            // Convert tokens to text
            var text = _tokenizer.Decode(tokens, skipSpecialTokens: true) ?? string.Empty;
            
            return new TranscriptionResult(text, detectedLanguage)
            {
                Duration = Duration(input)
            };
        }
        
        protected override TranscriptionResult TranscribeWithTimestamps(AudioClip audioClip)
        {
            var samples = ExtractSamples(audioClip);
            var melFeatures = ComputeMelSpectrogram(samples);
            var encoderOutput = EncodeAudio(melFeatures);
            var (tokens, detectedLanguage) = DecodeTokensWithTimestamps(encoderOutput);
            
            var result = new TranscriptionResult
            {
                Duration = Duration(audioClip),
                Language = detectedLanguage
            };
            
            // Parse tokens to extract text and timestamps
            ParseTokensWithTimestamps(tokens, result);
            
            return result;
        }
        
        protected override TranscriptionResult ForceAlign(string text, AudioClip audioClip)
        {
            // For force alignment, we constrain the decoder to produce the given text
            var samples = ExtractSamples(audioClip);
            var melFeatures = ComputeMelSpectrogram(samples);
            var encoderOutput = EncodeAudio(melFeatures);
            
            // Tokenize the target text
            var targetTokens = new List<int>(_tokenizer.Encode(text).Ids);
            
            // Force decode with timestamps
            var (alignedTokens, detectedLanguage) = ForceDecodeWithTimestamps(encoderOutput, targetTokens);
            
            var result = new TranscriptionResult
            {
                Text = text,
                Duration = Duration(audioClip),
                Language = detectedLanguage
            };
            
            ParseTokensWithTimestamps(alignedTokens, result);
            
            // If token-level alignment produced poor results, fall back to DTW
            if (result.WordTimestamps.Count == 0)
            {
                var dtwAligner = new DTWWordAligner(VerboseLogging);
                var transcribeResult = TranscribeWithTimestamps(audioClip);
                var transcribedWords = ExtractTranscribedWords(transcribeResult);
                return dtwAligner.AlignWords(text, transcribedWords, Duration(audioClip));
            }
            
            return result;
        }
        
        protected override TranscriptionResult ForceAlignWithVad(
            string text,
            AudioClip audioClip,
            float volumeThreshold = 0.02f,
            float minSegmentDuration = 0.1f,
            float maxSegmentGap = 0.3f)
        {
            var allSamples = ExtractSamples(audioClip);
            
            // Detect voice segments
            var voiceSegments = VoiceActivityDetector.DetectVoiceSegments(
                allSamples, WhisperConstants.SampleRate,
                volumeThreshold, minSegmentDuration, maxSegmentGap);
            
            if (VerboseLogging)
                Debug.Log($"[WhisperEngine] VAD detected {voiceSegments.Count} segments");
            
            if (voiceSegments.Count == 0)
                return ForceAlign(text, audioClip);
            
            // Transcribe each segment
            var transcribedWords = new List<TranscribedWord>();
            
            foreach (var segment in voiceSegments)
            {
                int startSample = Mathf.FloorToInt(segment.StartTime * WhisperConstants.SampleRate);
                int segmentLength = Mathf.CeilToInt(segment.Duration * WhisperConstants.SampleRate);
                
                // Pad to Whisper's expected length
                var segmentAudio = new float[WhisperConstants.MaxSamples];
                int copyLen = Mathf.Min(segmentLength, allSamples.Length - startSample);
                if (copyLen > 0)
                    System.Array.Copy(allSamples, startSample, segmentAudio, 0, copyLen);
                
                var melFeatures = ComputeMelSpectrogram(segmentAudio);
                var encoderOutput = EncodeAudio(melFeatures);
                var (tokens, _) = DecodeTokensWithTimestamps(encoderOutput);
                
                var segResult = new TranscriptionResult { Duration = segment.Duration };
                ParseTokensWithTimestamps(tokens, segResult);
                
                // Offset timestamps to absolute positions
                string segText = segResult.Text;
                if (!string.IsNullOrWhiteSpace(segText))
                {
                    string[] words = segText.Split(
                        new[] { ' ', '\t', '\n', '\r' },
                        System.StringSplitOptions.RemoveEmptyEntries);
                    
                    float wordDuration = segment.Duration / Mathf.Max(words.Length, 1);
                    for (int i = 0; i < words.Length; i++)
                    {
                        transcribedWords.Add(new TranscribedWord
                        {
                            Word = words[i],
                            StartTime = segment.StartTime + (i * wordDuration),
                            EndTime = segment.StartTime + ((i + 1) * wordDuration)
                        });
                    }
                }
            }
            
            // Use DTW to align transcribed words to expected text
            var dtwAligner = new DTWWordAligner(VerboseLogging);
            return dtwAligner.AlignWords(text, transcribedWords, Duration(audioClip));
        }
        
        /// <summary>
        /// Extract TranscribedWord list from a TranscriptionResult for DTW alignment.
        /// </summary>
        private static List<TranscribedWord> ExtractTranscribedWords(TranscriptionResult result)
        {
            var words = new List<TranscribedWord>();
            foreach (var wt in result.WordTimestamps)
            {
                words.Add(new TranscribedWord
                {
                    Word = wt.Word,
                    StartTime = wt.StartTime,
                    EndTime = wt.EndTime
                });
            }
            return words;
        }
        
        #region Audio Processing
        
        private float[] ExtractSamples(AudioClip clip)
        {
            if (ReferenceEquals(clip, _preparedClip) && _preparedSamples != null) return (float[])_preparedSamples.Clone();
            var samples = new float[clip.samples * clip.channels];
            clip.GetData(samples, 0);
            
            // Convert to mono
            if (clip.channels > 1)
            {
                var mono = new float[clip.samples];
                for (int i = 0; i < clip.samples; i++)
                {
                    float sum = 0;
                    for (int c = 0; c < clip.channels; c++)
                        sum += samples[i * clip.channels + c];
                    mono[i] = sum / clip.channels;
                }
                samples = mono;
            }
            
            // Resample to 16kHz if needed
            if (clip.frequency != WhisperConstants.SampleRate)
            {
                samples = Resample(samples, clip.frequency, WhisperConstants.SampleRate);
            }
            
            // Pad or truncate to max length
            if (samples.Length > WhisperConstants.MaxSamples)
            {
                Array.Resize(ref samples, WhisperConstants.MaxSamples);
            }
            else if (samples.Length < WhisperConstants.MaxSamples)
            {
                var padded = new float[WhisperConstants.MaxSamples];
                Array.Copy(samples, padded, samples.Length);
                samples = padded;
            }
            
            return samples;
        }
        protected override void ApplyRequest(AsrRequest request)
        {
            if (!string.IsNullOrWhiteSpace(request.Language)) _languageOverride = request.Language;
        }

        protected override void OnPrepareInput(AsrRequest request)
        {
            _preparedClip = request.Audio;
            _preparedSamples = ExtractSamples(request.Audio);
            _preparedDuration = request.Audio.length;
        }

        private float Duration(AudioClip clip)
        {
            return ReferenceEquals(clip, _preparedClip) ? _preparedDuration : clip.length;
        }
        
        private float[] Resample(float[] samples, int fromRate, int toRate)
        {
            double ratio = (double)toRate / fromRate;
            int newLength = (int)(samples.Length * ratio);
            var resampled = new float[newLength];
            
            for (int i = 0; i < newLength; i++)
            {
                double srcIndex = i / ratio;
                int srcI = (int)srcIndex;
                double frac = srcIndex - srcI;
                
                if (srcI + 1 < samples.Length)
                    resampled[i] = (float)(samples[srcI] * (1 - frac) + samples[srcI + 1] * frac);
                else
                    resampled[i] = samples[srcI];
            }
            
            return resampled;
        }
        
        #endregion
        
        #region Inference Steps
        
        private float[] ComputeMelSpectrogram(float[] samples)
        {
            var inputs = new Dictionary<string, OnnxTensor>
            {
                ["audio"] = OnnxTensor.FromArray(samples, new[] { 1, samples.Length }, "audio")
            };
            
            IReadOnlyDictionary<string, OnnxTensor> outputs = _melSession.Run(inputs);
            try
            {
                return RequiredOutput(outputs, "log_mel").AsFloatArray();
            }
            finally
            {
                DisposeTensors(inputs.Values);
                DisposeTensors(outputs.Values);
            }
        }
        
        private float[] EncodeAudio(float[] melFeatures)
        {
            if (melFeatures.Length % WhisperConstants.MelTimeSteps != 0)
                throw new InvalidOperationException($"Mel processor returned {melFeatures.Length} values, which is not divisible by {WhisperConstants.MelTimeSteps} time steps.");
            var shape = new[] { 1, melFeatures.Length / WhisperConstants.MelTimeSteps, WhisperConstants.MelTimeSteps };
            
            var inputs = new Dictionary<string, OnnxTensor>
            {
                [_encoderSession.InputNames.Contains("input_features") ? "input_features" : "mel"] =
                    OnnxTensor.FromArray(melFeatures, shape, "input_features")
            };
            
            IReadOnlyDictionary<string, OnnxTensor> outputs = _encoderSession.Run(inputs);
            try
            {
                return RequiredOutput(outputs, "last_hidden_state").AsFloatArray();
            }
            finally
            {
                DisposeTensors(inputs.Values);
                DisposeTensors(outputs.Values);
            }
        }
        
        private (List<int> tokens, string language) DecodeTokens(float[] encoderOutput)
        {
            var tokens = new List<int>();
            
            // Initial tokens: SOT, language, task, no_timestamps
            int languageToken = string.IsNullOrEmpty(_languageOverride) 
                ? WhisperConstants.GetLanguageToken("en") // Will be detected
                : WhisperConstants.GetLanguageToken(_languageOverride);
            
            int taskToken = _task == WhisperTask.Translate 
                ? WhisperConstants.Translate 
                : WhisperConstants.Transcribe;
            
            var promptTokens = new List<int>
            {
                WhisperConstants.StartOfTranscript,
                languageToken,
                taskToken,
                WhisperConstants.NoTimestamps
            };
            
            tokens.AddRange(promptTokens);
            
            string detectedLanguage = _languageOverride;
            
            using var decoderState = new DecoderState();
            for (int i = 0; i < _maxTokens; i++)
            {
                var nextToken = PredictNextToken(encoderOutput, tokens, decoderState);
                
                if (nextToken == WhisperConstants.EndOfText)
                    break;
                
                // Detect language from first generated token if auto-detecting
                if (string.IsNullOrEmpty(_languageOverride) && 
                    WhisperConstants.IsLanguageToken(nextToken))
                {
                    detectedLanguage = WhisperConstants.GetLanguageCode(nextToken);
                }
                
                tokens.Add(nextToken);
            }
            
            // Remove prompt tokens for final result
            tokens.RemoveRange(0, promptTokens.Count);
            
            return (tokens, detectedLanguage ?? "en");
        }
        
        private (List<int> tokens, string language) DecodeTokensWithTimestamps(float[] encoderOutput)
        {
            var tokens = new List<int>();
            
            int languageToken = string.IsNullOrEmpty(_languageOverride) 
                ? WhisperConstants.GetLanguageToken("en")
                : WhisperConstants.GetLanguageToken(_languageOverride);
            
            int taskToken = _task == WhisperTask.Translate 
                ? WhisperConstants.Translate 
                : WhisperConstants.Transcribe;
            
            var promptTokens = new List<int>
            {
                WhisperConstants.StartOfTranscript,
                languageToken,
                taskToken,
                WhisperConstants.StartTime // Enable timestamps
            };
            
            tokens.AddRange(promptTokens);
            
            string detectedLanguage = _languageOverride;
            
            using var decoderState = new DecoderState();
            for (int i = 0; i < _maxTokens; i++)
            {
                var nextToken = PredictNextToken(encoderOutput, tokens, decoderState);
                
                if (nextToken == WhisperConstants.EndOfText)
                    break;
                
                if (string.IsNullOrEmpty(_languageOverride) && 
                    WhisperConstants.IsLanguageToken(nextToken))
                {
                    detectedLanguage = WhisperConstants.GetLanguageCode(nextToken);
                }
                
                tokens.Add(nextToken);
            }
            
            return (tokens, detectedLanguage ?? "en");
        }
        
        private (List<int> tokens, string language) ForceDecodeWithTimestamps(
            float[] encoderOutput, 
            List<int> targetTokens)
        {
            var tokens = new List<int>();
            
            int languageToken = string.IsNullOrEmpty(_languageOverride) 
                ? WhisperConstants.GetLanguageToken("en")
                : WhisperConstants.GetLanguageToken(_languageOverride);
            
            var promptTokens = new List<int>
            {
                WhisperConstants.StartOfTranscript,
                languageToken,
                WhisperConstants.Transcribe,
                WhisperConstants.StartTime
            };
            
            tokens.AddRange(promptTokens);
            
            // Force-constrained decoding: feed target tokens but allow
            // the model to insert timestamp tokens between them
            int targetIdx = 0;
            
            using var decoderState = new DecoderState();
            for (int i = 0; i < _maxTokens && targetIdx <= targetTokens.Count; i++)
            {
                var nextToken = PredictNextToken(encoderOutput, tokens, decoderState);
                
                if (nextToken == WhisperConstants.EndOfText)
                    break;
                
                if (WhisperConstants.IsTimestampToken(nextToken))
                {
                    // Always accept timestamp tokens from the model
                    tokens.Add(nextToken);
                }
                else if (targetIdx < targetTokens.Count)
                {
                    // Force the target token instead of model's prediction
                    tokens.Add(targetTokens[targetIdx]);
                    targetIdx++;
                }
                else
                {
                    // All target tokens consumed, accept model output
                    tokens.Add(nextToken);
                }
            }
            
            return (tokens, _languageOverride ?? "en");
        }
        
        private int PredictNextToken(float[] encoderOutput, List<int> tokens, DecoderState state)
        {
            if (_decoderWithPastSession != null && state.Started)
                return PredictWithCache(tokens[tokens.Count - 1], state);

            var tokenArray = tokens.ToArray();
            const int encoderTimeSteps = 1500;
            if (encoderOutput.Length % encoderTimeSteps != 0)
                throw new InvalidOperationException($"Whisper encoder returned an unsupported output length: {encoderOutput.Length}.");
            var encoderShape = new[] { 1, encoderTimeSteps, encoderOutput.Length / encoderTimeSteps };
            
            var inputs = new Dictionary<string, OnnxTensor>
            {
                ["encoder_hidden_states"] = OnnxTensor.FromArray(encoderOutput, encoderShape, "encoder_hidden_states"),
                ["input_ids"] = OnnxTensor.FromArray(tokenArray, new[] { 1, tokens.Count }, "input_ids")
            };
            if (_decoderSession.InputNames.Contains("use_cache_branch"))
                inputs["use_cache_branch"] = OnnxTensor.FromArray(new[] { false }, new[] { 1 }, "use_cache_branch");
            int attentionHeads = encoderShape[2] / 64;
            foreach (string input in _decoderSession.InputNames.Where(name => name.StartsWith("past_key_values.", StringComparison.Ordinal)))
                inputs[input] = OnnxTensor.FromArray(Array.Empty<float>(), new[] { 1, attentionHeads, 0, 64 }, input);
            
            IReadOnlyDictionary<string, OnnxTensor> outputs = _decoderSession.Run(inputs);
            bool keepCache = _decoderWithPastSession != null;
            try
            {
                int nextToken = ArgMaxLastToken(RequiredOutput(outputs, "logits"));
                if (keepCache) state.Capture(outputs);
                return nextToken;
            }
            finally
            {
                DisposeTensors(inputs.Values);
                DisposeOutputs(outputs, keepCache);
            }
        }

        private int PredictWithCache(int token, DecoderState state)
        {
            OnnxTensor tokenInput = OnnxTensor.FromArray(new[] { token }, new[] { 1, 1 }, "input_ids");
            var inputs = new Dictionary<string, OnnxTensor> { ["input_ids"] = tokenInput };
            foreach (KeyValuePair<string, OnnxTensor> cache in state.Cache)
                inputs[cache.Key] = cache.Value;
            IReadOnlyDictionary<string, OnnxTensor> outputs = _decoderWithPastSession.Run(inputs);
            try
            {
                int nextToken = ArgMaxLastToken(RequiredOutput(outputs, "logits"));
                state.Capture(outputs);
                return nextToken;
            }
            finally
            {
                tokenInput.Dispose();
                DisposeOutputs(outputs, true);
            }
        }

        private static OnnxTensor RequiredOutput(IReadOnlyDictionary<string, OnnxTensor> outputs, string name)
        {
            if (!outputs.TryGetValue(name, out OnnxTensor tensor))
                throw new InvalidOperationException($"Whisper model did not produce required output '{name}'.");
            return tensor;
        }

        private static int ArgMaxLastToken(OnnxTensor logitsTensor)
        {
            float[] logits = logitsTensor.AsFloatArray();
            if (logitsTensor.Shape.Length == 0)
                throw new InvalidOperationException("Whisper logits have no shape.");
            int vocabSize = logitsTensor.Shape[logitsTensor.Shape.Length - 1];
            if (vocabSize <= 0 || logits.Length < vocabSize)
                throw new InvalidOperationException($"Whisper decoder returned an unsupported logits shape: [{string.Join(",", logitsTensor.Shape)}].");
            int lastTokenOffset = logits.Length - vocabSize;
            int bestToken = 0;
            float bestLogit = float.MinValue;
            for (int i = 0; i < vocabSize; i++)
            {
                float logit = logits[lastTokenOffset + i];
                if (logit > bestLogit)
                {
                    bestLogit = logit;
                    bestToken = i;
                }
            }
            return bestToken;
        }

        private static void DisposeOutputs(IReadOnlyDictionary<string, OnnxTensor> outputs, bool keepCache)
        {
            foreach (KeyValuePair<string, OnnxTensor> output in outputs)
                if (!keepCache || !output.Key.StartsWith("present.", StringComparison.Ordinal))
                    output.Value.Dispose();
        }

        private static void DisposeTensors(IEnumerable<OnnxTensor> tensors)
        {
            foreach (OnnxTensor tensor in tensors) tensor.Dispose();
        }

        private sealed class DecoderState : IDisposable
        {
            private readonly Dictionary<string, OnnxTensor> cache = new(StringComparer.Ordinal);
            public bool Started { get; private set; }
            public IReadOnlyDictionary<string, OnnxTensor> Cache => cache;

            public void Capture(IReadOnlyDictionary<string, OnnxTensor> outputs)
            {
                foreach (KeyValuePair<string, OnnxTensor> output in outputs)
                {
                    if (!output.Key.StartsWith("present.", StringComparison.Ordinal)) continue;
                    string inputName = "past_key_values." + output.Key.Substring("present.".Length);
                    if (cache.TryGetValue(inputName, out OnnxTensor previous)) previous.Dispose();
                    cache[inputName] = output.Value;
                }
                Started = true;
            }

            public void Dispose()
            {
                foreach (OnnxTensor tensor in cache.Values) tensor.Dispose();
                cache.Clear();
            }
        }
        
        #endregion
        
        #region Token Parsing
        
        private void ParseTokensWithTimestamps(List<int> tokens, TranscriptionResult result)
        {
            var textBuilder = new System.Text.StringBuilder();
            float currentStartTime = 0f;
            var currentWords = new List<string>();
            
            foreach (var token in tokens)
            {
                if (WhisperConstants.IsTimestampToken(token))
                {
                    float timestamp = WhisperConstants.TokenToTimestamp(token);
                    
                    if (currentWords.Count > 0)
                    {
                        // Create word timestamps for accumulated words
                        float wordDuration = (timestamp - currentStartTime) / currentWords.Count;
                        float wordStart = currentStartTime;
                        
                        foreach (var word in currentWords)
                        {
                            result.WordTimestamps.Add(new WordTimestamp(
                                word,
                                wordStart,
                                wordStart + wordDuration
                            ));
                            wordStart += wordDuration;
                        }
                        
                        currentWords.Clear();
                    }
                    
                    currentStartTime = timestamp;
                }
                else if (token < WhisperConstants.EndOfText)
                {
                    var text = _tokenizer.Decode(new[] { token }, skipSpecialTokens: true) ?? string.Empty;
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        textBuilder.Append(text);
                        currentWords.Add(text.Trim());
                    }
                }
            }
            
            result.Text = textBuilder.ToString().Trim();
        }
        
        #endregion
    }
    
    /// <summary>Whisper task type.</summary>
    public enum WhisperTask
    {
        /// <summary>Transcribe audio in its original language.</summary>
        Transcribe,
        
        /// <summary>Translate audio to English.</summary>
        Translate
    }
}
