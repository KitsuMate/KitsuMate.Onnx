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
        [Header("Model Set")]
        private readonly WhisperModelSet _modelSet;
        
        [Header("Transcription Settings")]
        [SerializeField, Tooltip("Language code (e.g., 'en', 'ja'). Leave empty for auto-detect.")]
        private string _languageOverride = "";
        
        [SerializeField, Tooltip("Task: transcribe or translate to English")]
        private WhisperTask _task = WhisperTask.Transcribe;
        
        [SerializeField, Tooltip("Maximum number of tokens to generate")]
        private int _maxTokens = 224;
        private AudioClip _preparedClip;
        private float[] _preparedSamples;
        
        // Runtime state
        private IOnnxSession _melSession;
        private IOnnxSession _encoderSession;
        private IOnnxSession _decoderSession;
        private Tokenizer _tokenizer;
        
        /// <summary>Model set containing Whisper models.</summary>
        public WhisperEngineRuntime(WhisperModelSet modelSet, string languageOverride, WhisperTask task, int maxTokens, bool verbose)
        {
            _modelSet = modelSet;
            _languageOverride = languageOverride ?? string.Empty;
            _task = task;
            _maxTokens = maxTokens;
            VerboseLogging = verbose;
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
            if (_modelSet == null || !_modelSet.IsComplete)
                throw new InvalidOperationException("WhisperModelSet is not assigned or incomplete");

            _tokenizer = Tokenizer.FromTokenizerJson(_modelSet.TokenizerJson.bytes);
        }
        
        protected override void OnLoadBackground(OnnxBackend backend)
        {
            if (_modelSet == null || !_modelSet.IsComplete)
                throw new InvalidOperationException("WhisperModelSet is not assigned or incomplete");
            
            // Create sessions
            _melSession = backend.CreateSession(_modelSet.MelProcessor);
            _encoderSession = backend.CreateSession(_modelSet.Encoder);
            _decoderSession = backend.CreateSession(_modelSet.Decoder);
            
            if (VerboseLogging)
                Debug.Log($"[WhisperEngine] Loaded {_modelSet.DisplayName}");
        }
        
        protected override void OnUnload()
        {
            _melSession?.Dispose();
            _encoderSession?.Dispose();
            _decoderSession?.Dispose();
            
            _melSession = null;
            _encoderSession = null;
            _decoderSession = null;
            _tokenizer = null;
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
                Duration = input.length
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
                Duration = audioClip.length,
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
                Duration = audioClip.length,
                Language = detectedLanguage
            };
            
            ParseTokensWithTimestamps(alignedTokens, result);
            
            // If token-level alignment produced poor results, fall back to DTW
            if (result.WordTimestamps.Count == 0)
            {
                var dtwAligner = new DTWWordAligner(VerboseLogging);
                var transcribeResult = TranscribeWithTimestamps(audioClip);
                var transcribedWords = ExtractTranscribedWords(transcribeResult);
                return dtwAligner.AlignWords(text, transcribedWords, audioClip.length);
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
            return dtwAligner.AlignWords(text, transcribedWords, audioClip.length);
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
            
            var outputs = _melSession.Run(inputs);
            var melTensor = outputs.Values.First();
            
            return melTensor.AsFloatArray();
        }
        
        private float[] EncodeAudio(float[] melFeatures)
        {
            // Shape: [1, mel_bins, time_steps]
            var shape = new[] { 1, WhisperConstants.MelBins, WhisperConstants.MelTimeSteps };
            
            var inputs = new Dictionary<string, OnnxTensor>
            {
                ["mel"] = OnnxTensor.FromArray(melFeatures, shape, "mel")
            };
            
            var outputs = _encoderSession.Run(inputs);
            var encoderOutput = outputs.Values.First();
            
            return encoderOutput.AsFloatArray();
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
            
            // Greedy decoding
            for (int i = 0; i < _maxTokens; i++)
            {
                var nextToken = PredictNextToken(encoderOutput, tokens);
                
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
            
            for (int i = 0; i < _maxTokens; i++)
            {
                var nextToken = PredictNextToken(encoderOutput, tokens);
                
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
            
            for (int i = 0; i < _maxTokens && targetIdx <= targetTokens.Count; i++)
            {
                var nextToken = PredictNextToken(encoderOutput, tokens);
                
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
        
        private int PredictNextToken(float[] encoderOutput, List<int> tokens)
        {
            // Prepare decoder inputs
            var tokenArray = tokens.Select(t => (long)t).ToArray();
            var encoderShape = new[] { 1, 1500, 512 }; // Typical whisper encoder output shape
            
            var inputs = new Dictionary<string, OnnxTensor>
            {
                ["encoder_hidden_states"] = OnnxTensor.FromArray(encoderOutput, encoderShape, "encoder_hidden_states"),
                ["input_ids"] = OnnxTensor.FromArray(tokenArray, new[] { 1, tokens.Count }, "input_ids")
            };
            
            var outputs = _decoderSession.Run(inputs);
            var logits = outputs.Values.First().AsFloatArray();
            
            // Get the last token's logits and find argmax
            int vocabSize = WhisperConstants.VocabularySize;
            int lastTokenOffset = (tokens.Count - 1) * vocabSize;
            
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
