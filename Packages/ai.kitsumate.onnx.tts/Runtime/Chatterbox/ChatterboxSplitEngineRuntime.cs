using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using KitsuMate.Tokenizers;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    /// <summary>V3 static-cache language model and split flow decoder.</summary>
    public sealed class ChatterboxSplitEngineRuntime : TtsEngineRuntime
    {
        /// <summary>
        /// Language model positions allocated once per runtime: conditioning, text, and generated tokens.
        /// The default 500-token limit leaves room for about 490 text and conditioning positions.
        /// </summary>
        private const int CacheCapacity = 1024;

        private readonly ChatterboxSplitModelSet modelSet;
        private readonly int voiceCacheCapacity;
        private IOnnxSession encoder, tokenEmbedding, languageModel, flowPrepare, flowStep, vocoder;
        private ChatterboxStaticLanguageModel staticLanguageModel;
        private Tokenizer tokenizer;
        private LanguagePreprocessor languagePreprocessor;
        private bool multilingual;
        private float[] voiceSamples;
        private EntityId voiceClip;
        private ChatterboxVoiceCache voiceCache;

        public ChatterboxSplitEngineRuntime(ChatterboxSplitModelSet modelSet, int voiceCacheCapacity, bool verbose)
        {
            this.modelSet = modelSet;
            this.voiceCacheCapacity = voiceCacheCapacity;
            VerboseLogging = verbose;
        }

        public override int OutputSampleRate => ChatterboxConstants.SampleRate;
        public override bool IsMultilingual => multilingual;
        public override string[] SupportedLanguages => multilingual
            ? ChatterboxConstants.SupportedLanguages.Keys.ToArray() : Array.Empty<string>();

        protected override void OnLoadMainThread(OnnxBackend backend)
        {
            if (modelSet == null || !modelSet.IsComplete)
                throw new InvalidOperationException("Chatterbox split model set is incomplete.");
            tokenizer = Tokenizer.FromTokenizerJson(modelSet.Tokenizer.bytes);
            multilingual = modelSet.Tokenizer.text.Contains("\"[ko]\"") ||
                modelSet.Tokenizer.text.Contains("\"[zh]\"");
            languagePreprocessor = new LanguagePreprocessor(
                modelSet.CangjieMapping?.text,
                modelSet.JapaneseReadings?.text,
                modelSet.RussianStress?.text,
                modelSet.ChineseWords?.text);
        }

        protected override void OnLoadBackground(OnnxBackend backend)
        {
            var options = new OnnxSessionOptions
                { IntraOpThreads = Math.Min(4, Environment.ProcessorCount), InterOpThreads = 1 };
            try
            {
                encoder = backend.CreateSession(modelSet.SpeechEncoder, options);
                // Embedding a token is a table lookup, cheaper on the CPU than a device round trip.
                tokenEmbedding = backend.CreateSession(modelSet.TokenEmbedding, new OnnxSessionOptions
                {
                    IntraOpThreads = 1, InterOpThreads = 1, Providers = new[] { OnnxExecutionProvider.Cpu }
                });
                languageModel = backend.CreateSession(modelSet.LanguageModel, new OnnxSessionOptions
                {
                    IntraOpThreads = options.IntraOpThreads, InterOpThreads = 1, EnableGraphCapture = true
                });
                flowPrepare = backend.CreateSession(modelSet.FlowPrepare, options);
                flowStep = backend.CreateSession(modelSet.FlowStep, options);
                vocoder = backend.CreateSession(modelSet.Vocoder, options);
                foreach (var session in Sessions()) session.VerboseLogging = VerboseLogging;
                staticLanguageModel = new ChatterboxStaticLanguageModel(tokenEmbedding, languageModel, CacheCapacity);
                voiceCache = voiceCacheCapacity > 0 ? new ChatterboxVoiceCache(voiceCacheCapacity) : null;
            }
            catch { OnUnload(); throw; }
        }

        protected override void OnUnload()
        {
            // The cache tensors belong to the language model session and go first.
            staticLanguageModel?.Dispose(); staticLanguageModel = null;
            encoder?.Dispose(); encoder = null;
            tokenEmbedding?.Dispose(); tokenEmbedding = null;
            languageModel?.Dispose(); languageModel = null;
            flowPrepare?.Dispose(); flowPrepare = null;
            flowStep?.Dispose(); flowStep = null;
            vocoder?.Dispose(); vocoder = null;
            tokenizer = null;
            languagePreprocessor = null;
            voiceSamples = null;
            voiceCache?.Dispose(); voiceCache = null;
            multilingual = false;
        }

        private IEnumerable<IOnnxSession> Sessions()
        {
            yield return encoder; yield return tokenEmbedding; yield return languageModel; yield return flowPrepare;
            yield return flowStep; yield return vocoder;
        }

        private static int IndexOfOutput(IOnnxSession session, string name)
        {
            for (int i = 0; i < session.OutputNames.Count; i++)
                if (session.OutputNames[i] == name) return i;
            throw new OnnxModelContractException($"The graph does not declare output '{name}'.");
        }

        protected override void OnPrepareInput(TtsRequest input)
        {
            AudioClip clip = input.VoiceReference ?? modelSet.DefaultVoice;
            if (clip == null) throw new InvalidOperationException("A reference voice is required.");
            voiceClip = clip.GetEntityId();
            // Reading and resampling the clip is main-thread work; an already encoded voice needs neither.
            voiceSamples = voiceCache != null && voiceCache.Contains(voiceClip)
                ? null
                : ChatterboxEngineRuntime.PrepareVoiceSamples(clip);
        }

        protected override TtsResult OnRun(TtsRequest input, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ChatterboxGenerationConfig generation = input.Chatterbox ?? modelSet.Generation;
            generation.Validate();
            string language = string.IsNullOrWhiteSpace(input.LanguageId) ? "en" : input.LanguageId.ToLowerInvariant();
            if (!ChatterboxConstants.SupportedLanguages.ContainsKey(language))
                throw new ArgumentException($"Unsupported Chatterbox language: {language}");
            string text = ChatterboxEngineRuntime.PrepareV3Text(input.Text);
            if (multilingual) text = $"[{language}]" + languagePreprocessor.ProcessV3(text, language);
            text = text.Replace(" ", "[SPACE]");
            long[] ids = tokenizer.Encode(text, addSpecialTokens: true).Ids.Select(id => (long)id).ToArray();
            long[] positions = ids.Select((id, index) => id >= ChatterboxConstants.StartSpeechToken
                ? 0L : Math.Max(0, index - 1)).ToArray();

            var total = VerboseLogging ? Stopwatch.StartNew() : null;
            var phase = VerboseLogging ? Stopwatch.StartNew() : null;
            bool encoded = voiceSamples != null;
            (ChatterboxVoiceConditioning voice, bool owned) = EncodeVoice();
            Log(phase, encoded ? "speech encoder" : "voice cache hit");
            try
            {
                var generated = Generate(ids, positions, voice.Features, input, generation, cancellationToken);
                int speechCount = generated.Count;
                Log(phase, $"generation ({speechCount} speech tokens)");
                if (speechCount == 0)
                    throw new InvalidOperationException("Chatterbox generated no speech tokens.");
                long[] prompt = voice.Tokens.AsLongArray();
                var speechTokens = new long[prompt.Length + speechCount];
                Array.Copy(prompt, speechTokens, prompt.Length);
                for (int i = 0; i < speechCount; i++) speechTokens[prompt.Length + i] = generated[i];
                float[] waveform = Decode(speechTokens, voice.Speaker, voice.SpeakerFeatures,
                    generation.Seed, cancellationToken);
                if (total != null)
                    Debug.Log($"[ChatterboxSplit] total {total.ElapsedMilliseconds}ms for " +
                              $"{waveform.Length / (float)ChatterboxConstants.SampleRate:F2}s audio");
                return new TtsResult { Samples = waveform, SampleRate = OutputSampleRate,
                    TokensGenerated = speechCount };
            }
            finally { if (owned) voice.Dispose(); }
        }

        /// <summary>
        /// Returns the conditioning for the requested voice, encoding it only when the cache misses.
        /// The second value says whether the caller owns the result; cached entries belong to the cache.
        /// </summary>
        private (ChatterboxVoiceConditioning Voice, bool Owned) EncodeVoice()
        {
            if (voiceCache != null && voiceCache.TryGet(voiceClip, out ChatterboxVoiceConditioning cached))
                return (cached, false);

            using var audio = OnnxTensor.FromArray(voiceSamples, new[] { 1, voiceSamples.Length });
            var outputs = encoder.Run(new Dictionary<string, OnnxTensor> { ["audio_values"] = audio });
            var voice = new ChatterboxVoiceConditioning(outputs["audio_features"], outputs["audio_tokens"],
                outputs["speaker_embeddings"], outputs["speaker_features"]);
            if (voiceCache == null) return (voice, true);
            voiceCache.Put(voiceClip, voice);
            return (voice, false);
        }

        private List<int> Generate(long[] ids, long[] positions, OnnxTensor conditioning,
            TtsRequest request, ChatterboxGenerationConfig generation, CancellationToken cancellationToken)
        {
            int requested = request.MaxNewTokens > 0 ? request.MaxNewTokens : ChatterboxConstants.DefaultMaxNewTokens;
            var random = new System.Random(generation.Seed);
            var repetition = new RepetitionPenaltyProcessor(request.RepetitionPenalty > 0
                ? request.RepetitionPenalty : ChatterboxConstants.DefaultRepetitionPenalty);
            var history = new List<int>(requested + 1) { ChatterboxConstants.StartSpeechToken };
            int vocabulary = ChatterboxConstants.SpeechLogitCount;
            using var batchConditioning = OnnxTensor.FromArray(
                Repeat(conditioning.AsFloatArray(), ChatterboxStaticLanguageModel.Batch),
                new[] { ChatterboxStaticLanguageModel.Batch, conditioning.Shape[1], conditioning.Shape[2] });
            float[] logits = staticLanguageModel.Prefill(ids, positions, batchConditioning, request.Exaggeration);
            // Generation stops at the cache end; a text that long already produces about 20 seconds of speech.
            int limit = Math.Min(requested, staticLanguageModel.Capacity - ids.Length - conditioning.Shape[1]);
            for (int step = 0; step < limit; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (step > 0) logits = staticLanguageModel.Step(history[^1], step, request.Exaggeration);
                float[] scores = ChatterboxSampling.Combine(logits, vocabulary, ChatterboxStaticLanguageModel.Batch,
                    generation.Guidance);
                repetition.Apply(history, scores);
                int next = ChatterboxSampling.Sample(scores, generation, random);
                history.Add(next);
                if (next == ChatterboxConstants.StopSpeechToken) break;
            }
            return history.Skip(1).Where(token => token >= 0 && token < ChatterboxConstants.StartSpeechToken)
                .ToList();
        }

        private float[] Decode(long[] speechTokens, OnnxTensor speaker, OnnxTensor features,
            int seed, CancellationToken cancellationToken)
        {
            var phase = VerboseLogging ? Stopwatch.StartNew() : null;
            using var tokens = OnnxTensor.FromArray(speechTokens, new[] { 1, speechTokens.Length });
            var prepared = flowPrepare.Run(new Dictionary<string, OnnxTensor>
            {
                ["speech_tokens"] = tokens, ["speaker_embeddings"] = speaker, ["speaker_features"] = features
            });
            Log(phase, "flow prepare");
            OnnxTensor x = null;
            try
            {
                int[] noiseShape = prepared["x"].Shape;
                x = OnnxTensor.FromArray(GenerateFlowNoise(noiseShape, seed), noiseShape);
                x = SolveFlow(x, prepared, cancellationToken);
                Log(phase, $"flow steps ({modelSet.FlowSteps})");
                using var prefix = OnnxTensor.FromArray(prepared["prefix"].AsLongArray(), new[] { 1 });
                var outputWaveform = vocoder.Run(new Dictionary<string, OnnxTensor>
                    { ["x"] = x, ["prefix"] = prefix });
                try
                {
                    Log(phase, "vocoder");
                    float[] samples = outputWaveform["waveform"].AsFloatArray();
                    if (samples.Any(sample => !float.IsFinite(sample)))
                        throw new InvalidOperationException("Split decoder produced nonfinite audio.");
                    return samples;
                }
                finally { DisposeAll(outputWaveform); }
            }
            finally
            {
                x?.Dispose();
                DisposeAll(prepared);
            }
        }

        /// <summary>
        /// Runs the fixed Euler schedule. The evolving mel is the only value that changes between
        /// steps, so it stays on the inference device and only the final state is read back.
        /// </summary>
        private OnnxTensor SolveFlow(OnnxTensor noise, IReadOnlyDictionary<string, OnnxTensor> prepared,
            CancellationToken cancellationToken)
        {
            int steps = modelSet.FlowSteps;
            using var guidance = OnnxTensor.FromArray(new[] { modelSet.FlowGuidance }, new[] { 1 });
            var constants = new[]
            {
                new OnnxNamedValue("mask", prepared["mask"]), new OnnxNamedValue("mu", prepared["mu"]),
                new OnnxNamedValue("speakers", prepared["speakers"]), new OnnxNamedValue("cond", prepared["cond"]),
                new OnnxNamedValue("guidance", guidance)
            };

            List<OnnxNamedValue> StepInputs(int index)
            {
                float time = FlowTime(index, steps);
                var inputs = new List<OnnxNamedValue>(8)
                {
                    new("time", OnnxTensor.FromArray(new[] { time }, new[] { 1 })),
                    new("delta", OnnxTensor.FromArray(new[] { FlowTime(index + 1, steps) - time }, new[] { 1 }))
                };
                inputs.AddRange(constants);
                return inputs;
            }

            if (flowStep is not IOnnxDeviceSession device)
            {
                OnnxTensor x = noise;
                for (int i = 0; i < steps; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var inputs = StepInputs(i);
                    inputs.Add(new OnnxNamedValue("x", x));
                    OnnxTensor next = flowStep.Run(inputs)["next_x"];
                    x.Dispose();
                    x = next;
                }
                return x;
            }

            int nextIndex = IndexOfOutput(flowStep, "next_x");
            IDeviceTensor resident = null;
            try
            {
                for (int i = 0; i < steps; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var inputs = StepInputs(i);
                    if (resident == null) inputs.Add(new OnnxNamedValue("x", noise));
                    var outputs = device.RunOnDevice(inputs,
                        resident == null ? Array.Empty<IDeviceTensor>() : new[] { resident },
                        Array.Empty<string>());
                    for (int output = 0; output < outputs.Count; output++)
                        if (output != nextIndex) outputs[output].Dispose();
                    resident?.Dispose();
                    resident = outputs[nextIndex];
                    resident.Name = "x";
                }
                if (resident == null) return noise;
                // A host-backed device tensor hands back its own buffer, so re-wrap the data before
                // the finally below releases the tensor along with the device handle.
                OnnxTensor readback = resident.ToCpu();
                OnnxTensor solved = OnnxTensor.FromArray(readback.AsFloatArray(), readback.Shape);
                noise.Dispose();
                return solved;
            }
            finally { resident?.Dispose(); }
        }

        internal static float[] GenerateFlowNoise(int[] shape, int seed)
        {
            int count = 1;
            foreach (int dimension in shape)
            {
                if (dimension <= 0) throw new ArgumentException("Flow noise shape must be positive.", nameof(shape));
                count = checked(count * dimension);
            }
            var result = new float[count];
            var random = new System.Random(seed);
            for (int i = 0; i < count; i += 2)
            {
                double radius = Math.Sqrt(-2d * Math.Log(1d - random.NextDouble()));
                double angle = 2d * Math.PI * random.NextDouble();
                result[i] = (float)(radius * Math.Cos(angle));
                if (i + 1 < count) result[i + 1] = (float)(radius * Math.Sin(angle));
            }
            return result;
        }

        internal static float FlowTime(int step, int steps) =>
            1f - (float)Math.Cos((double)step / steps * Math.PI * 0.5);

        private static T[] Repeat<T>(T[] source, int count)
        {
            if (count == 1) return source;
            var result = new T[source.Length * count];
            for (int i = 0; i < count; i++) Array.Copy(source, 0, result, i * source.Length, source.Length);
            return result;
        }

        private void Log(Stopwatch phase, string label)
        {
            if (phase == null) return;
            Debug.Log($"[ChatterboxSplit] {label}: {phase.ElapsedMilliseconds}ms");
            phase.Restart();
        }

        private static void DisposeAll(IReadOnlyDictionary<string, OnnxTensor> values)
        {
            foreach (var tensor in values.Values) tensor?.Dispose();
        }
    }
}
