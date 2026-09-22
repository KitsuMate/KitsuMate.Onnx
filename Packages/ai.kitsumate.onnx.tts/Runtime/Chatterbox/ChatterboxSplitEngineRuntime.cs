using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using KitsuMate.Tokenizers;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    /// <summary>V3 merged language model and split flow decoder.</summary>
    public sealed class ChatterboxSplitEngineRuntime : TtsEngineRuntime
    {
        private readonly ChatterboxSplitModelSet modelSet;
        private IOnnxSession encoder, languageModel, flowPrepare, flowStep, vocoder;
        private Tokenizer tokenizer;
        private LanguagePreprocessor languagePreprocessor;
        private bool multilingual;
        private float[] voiceSamples;
        private string[] pastNames;
        private Dictionary<string, string> presentToPast;

        public ChatterboxSplitEngineRuntime(ChatterboxSplitModelSet modelSet, bool verbose)
        {
            this.modelSet = modelSet;
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
                languageModel = backend.CreateSession(modelSet.EmbeddingLanguageModel, options);
                flowPrepare = backend.CreateSession(modelSet.FlowPrepare, options);
                flowStep = backend.CreateSession(modelSet.FlowStep, options);
                vocoder = backend.CreateSession(modelSet.Vocoder, options);
                foreach (var session in Sessions()) session.VerboseLogging = VerboseLogging;
                pastNames = languageModel.InputNames.Where(name =>
                    name.StartsWith("past_key_values.", StringComparison.Ordinal)).ToArray();
                presentToPast = languageModel.OutputNames.Where(name =>
                    name.StartsWith("present.", StringComparison.Ordinal)).ToDictionary(name => name,
                    name => "past_key_values." + name.Substring("present.".Length), StringComparer.Ordinal);
                if (pastNames.Length == 0 || presentToPast.Count != pastNames.Length)
                    throw new OnnxModelContractException("The merged language model has an incomplete KV cache contract.");
            }
            catch { OnUnload(); throw; }
        }

        protected override void OnUnload()
        {
            encoder?.Dispose(); encoder = null;
            languageModel?.Dispose(); languageModel = null;
            flowPrepare?.Dispose(); flowPrepare = null;
            flowStep?.Dispose(); flowStep = null;
            vocoder?.Dispose(); vocoder = null;
            tokenizer = null;
            languagePreprocessor = null;
            voiceSamples = null;
            pastNames = null;
            presentToPast = null;
            multilingual = false;
        }

        private IEnumerable<IOnnxSession> Sessions()
        {
            yield return encoder; yield return languageModel; yield return flowPrepare;
            yield return flowStep; yield return vocoder;
        }

        protected override void OnPrepareInput(TtsRequest input)
        {
            AudioClip clip = input.VoiceReference ?? modelSet.DefaultVoice;
            if (clip == null) throw new InvalidOperationException("A reference voice is required.");
            voiceSamples = ChatterboxEngineRuntime.PrepareVoiceSamples(clip);
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

            using var voiceInput = OnnxTensor.FromArray(voiceSamples, new[] { 1, voiceSamples.Length });
            var voice = encoder.Run(new Dictionary<string, OnnxTensor> { ["audio_values"] = voiceInput });
            try
            {
                var generated = Generate(ids, positions, voice["audio_features"], input, generation,
                    cancellationToken);
                int speechCount = generated.Count;
                if (speechCount == 0)
                    throw new InvalidOperationException("Chatterbox generated no speech tokens.");
                long[] prompt = voice["audio_tokens"].AsLongArray();
                var speechTokens = new long[prompt.Length + speechCount];
                Array.Copy(prompt, speechTokens, prompt.Length);
                for (int i = 0; i < speechCount; i++) speechTokens[prompt.Length + i] = generated[i];
                float[] waveform = Decode(speechTokens, voice["speaker_embeddings"],
                    voice["speaker_features"], generation.Seed, cancellationToken);
                return new TtsResult { Samples = waveform, SampleRate = OutputSampleRate,
                    TokensGenerated = speechCount };
            }
            finally { DisposeAll(voice); }
        }

        private List<int> Generate(long[] initialIds, long[] initialPositions, OnnxTensor conditioning,
            TtsRequest request, ChatterboxGenerationConfig generation, CancellationToken cancellationToken)
        {
            int batch = generation.Guidance > 0 ? 2 : 1;
            int limit = request.MaxNewTokens > 0 ? request.MaxNewTokens : ChatterboxConstants.DefaultMaxNewTokens;
            var random = new System.Random(generation.Seed);
            var repetition = new RepetitionPenaltyProcessor(request.RepetitionPenalty > 0
                ? request.RepetitionPenalty : ChatterboxConstants.DefaultRepetitionPenalty);
            var history = new List<int>(limit + 1) { ChatterboxConstants.StartSpeechToken };
            var cpuCache = languageModel is IOnnxDeviceSession ? null : new Dictionary<string, OnnxTensor>();
            var device = languageModel as IOnnxDeviceSession;
            var deviceCache = device == null ? null : new IDeviceTensor[pastNames.Length];
            int totalLength = 0;
            try
            {
                for (int step = 0; step < limit; step++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    long[] ids = step == 0 ? initialIds : new[] { (long)history[^1] };
                    long[] tokenPositions = step == 0 ? initialPositions : new[] { (long)step };
                    int conditioningLength = step == 0 ? conditioning.Shape[1] : 0;
                    int previousLength = totalLength;
                    totalLength += ids.Length + conditioningLength;
                    var inputs = new List<OnnxNamedValue>();
                    var allocated = new List<OnnxTensor>();
                    void Add(string name, OnnxTensor tensor) { inputs.Add(new OnnxNamedValue(name, tensor)); allocated.Add(tensor); }
                    try
                    {
                        Add("input_ids", OnnxTensor.FromArray(Repeat(ids, batch), new[] { batch, ids.Length }));
                        Add("token_position_ids", OnnxTensor.FromArray(Repeat(tokenPositions, batch),
                            new[] { batch, tokenPositions.Length }));
                        Add("exaggeration", OnnxTensor.FromArray(Enumerable.Repeat(request.Exaggeration, batch).ToArray(),
                            new[] { batch }));
                        Add("text_conditioning", OnnxTensor.FromArray(batch == 2 ? new[] { 1f, 0f } : new[] { 1f },
                            new[] { batch }));
                        float[] cond = step == 0 ? Repeat(conditioning.AsFloatArray(), batch) : Array.Empty<float>();
                        Add("conditioning", OnnxTensor.FromArray(cond, new[] { batch, conditioningLength, 1024 }));
                        Add("attention_mask", OnnxTensor.FromArray(Enumerable.Repeat(1L, batch * totalLength).ToArray(),
                            new[] { batch, totalLength }));
                        long[] lmPositions = Enumerable.Range(previousLength, totalLength - previousLength)
                            .Select(index => (long)index).ToArray();
                        Add("lm_position_ids", OnnxTensor.FromArray(Repeat(lmPositions, batch),
                            new[] { batch, lmPositions.Length }));

                        float[] logits;
                        int vocabulary;
                        if (device != null)
                        {
                            if (step == 0)
                                foreach (string name in pastNames)
                                    Add(name, OnnxTensor.FromArray(Array.Empty<float>(),
                                        new[] { batch, 16, 0, ChatterboxConstants.HeadDim }));
                            var outputs = device.RunOnDevice(inputs, step == 0
                                ? Array.Empty<IDeviceTensor>() : deviceCache, new[] { "logits" });
                            int logitsIndex = languageModel.OutputNames.ToList().IndexOf("logits");
                            using (var tensor = outputs[logitsIndex].ToCpu())
                            {
                                logits = tensor.AsFloatArray();
                                vocabulary = tensor.Shape[^1];
                            }
                            for (int i = 0; i < outputs.Count; i++)
                            {
                                if (i == logitsIndex) { outputs[i].Dispose(); continue; }
                                string outputName = languageModel.OutputNames[i];
                                if (!presentToPast.TryGetValue(outputName, out string pastName))
                                { outputs[i].Dispose(); continue; }
                                int cacheIndex = Array.IndexOf(pastNames, pastName);
                                deviceCache[cacheIndex]?.Dispose();
                                outputs[i].Name = pastName;
                                deviceCache[cacheIndex] = outputs[i];
                            }
                        }
                        else
                        {
                            foreach (string name in pastNames)
                            {
                                if (!cpuCache.TryGetValue(name, out OnnxTensor cache))
                                {
                                    cache = OnnxTensor.FromArray(Array.Empty<float>(),
                                        new[] { batch, 16, 0, ChatterboxConstants.HeadDim });
                                    cpuCache[name] = cache;
                                }
                                inputs.Add(new OnnxNamedValue(name, cache));
                            }
                            var outputs = languageModel.Run(inputs);
                            logits = outputs["logits"].AsFloatArray();
                            vocabulary = outputs["logits"].Shape[^1];
                            foreach (var pair in presentToPast)
                            {
                                cpuCache[pair.Value]?.Dispose();
                                cpuCache[pair.Value] = outputs[pair.Key];
                            }
                            outputs["logits"].Dispose();
                        }
                        float[] scores = ChatterboxSampling.Combine(logits, vocabulary, batch, generation.Guidance);
                        repetition.Apply(history, scores);
                        int next = ChatterboxSampling.Sample(scores, generation, random);
                        history.Add(next);
                        if (next == ChatterboxConstants.StopSpeechToken) break;
                    }
                    finally { foreach (var tensor in allocated) tensor.Dispose(); }
                }
            }
            finally
            {
                if (deviceCache != null) foreach (var tensor in deviceCache) tensor?.Dispose();
                if (cpuCache != null) foreach (var tensor in cpuCache.Values) tensor?.Dispose();
            }
            return history.Skip(1).Where(token => token >= 0 && token < ChatterboxConstants.StartSpeechToken)
                .ToList();
        }

        private float[] Decode(long[] speechTokens, OnnxTensor speaker, OnnxTensor features,
            int seed, CancellationToken cancellationToken)
        {
            using var tokens = OnnxTensor.FromArray(speechTokens, new[] { 1, speechTokens.Length });
            var prepared = flowPrepare.Run(new Dictionary<string, OnnxTensor>
            {
                ["speech_tokens"] = tokens, ["speaker_embeddings"] = speaker, ["speaker_features"] = features
            });
            OnnxTensor x = null;
            try
            {
                int[] noiseShape = prepared["x"].Shape;
                x = OnnxTensor.FromArray(GenerateFlowNoise(noiseShape, seed), noiseShape);
                int steps = modelSet.FlowSteps;
                for (int i = 0; i < steps; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    float time = FlowTime(i, steps);
                    float delta = FlowTime(i + 1, steps) - time;
                    using var timeTensor = OnnxTensor.FromArray(new[] { time }, new[] { 1 });
                    using var deltaTensor = OnnxTensor.FromArray(new[] { delta }, new[] { 1 });
                    using var guidanceTensor = OnnxTensor.FromArray(new[] { modelSet.FlowGuidance }, new[] { 1 });
                    var output = flowStep.Run(new Dictionary<string, OnnxTensor>
                    {
                        ["x"] = x, ["mask"] = prepared["mask"], ["mu"] = prepared["mu"],
                        ["speakers"] = prepared["speakers"], ["cond"] = prepared["cond"],
                        ["time"] = timeTensor, ["delta"] = deltaTensor, ["guidance"] = guidanceTensor
                    });
                    var next = output["next_x"];
                    x.Dispose();
                    x = next;
                }
                using var prefix = OnnxTensor.FromArray(prepared["prefix"].AsLongArray(), new[] { 1 });
                var outputWaveform = vocoder.Run(new Dictionary<string, OnnxTensor>
                    { ["x"] = x, ["prefix"] = prefix });
                try
                {
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
            var result = new T[source.Length * count];
            for (int i = 0; i < count; i++) Array.Copy(source, 0, result, i * source.Length, source.Length);
            return result;
        }

        private static void DisposeAll(IReadOnlyDictionary<string, OnnxTensor> values)
        {
            foreach (var tensor in values.Values) tensor?.Dispose();
        }
    }
}
