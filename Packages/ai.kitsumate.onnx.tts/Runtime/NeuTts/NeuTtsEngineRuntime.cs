using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using KitsuMate.Tokenizers;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.NeuTts
{
    public sealed class NeuTtsEngineRuntime : TtsEngineRuntime
    {
        private readonly NeuTtsModelSet modelSet;
        private readonly NeuTtsGenerationConfig defaults;
        private IOnnxSession backbone, codec;
        private Tokenizer tokenizer;
        private NeuTtsMetadata metadata;
        private Dictionary<int, int> speechCodes;
        public NeuTtsEngineRuntime(NeuTtsModelSet set, NeuTtsGenerationConfig generation, bool verbose)
        { modelSet = set; defaults = generation?.Clone() ?? new(); VerboseLogging = verbose; }
        public override int OutputSampleRate => 24000;
        public override bool IsMultilingual => false;
        public override string[] SupportedLanguages => Array.Empty<string>();
        public OnnxSessionDiagnostics PrimarySessionDiagnostics => backbone?.Diagnostics;
        public IReadOnlyList<OnnxSessionDiagnostics> SessionDiagnostics => new[] { backbone?.Diagnostics, codec?.Diagnostics };

        protected override void OnLoadMainThread(OnnxBackend backend)
        {
            if (modelSet == null || !modelSet.IsComplete) throw new InvalidOperationException("NeuTTS model set is incomplete.");
            metadata = JsonUtility.FromJson<NeuTtsMetadata>(modelSet.Metadata.text);
            metadata.Validate();
            tokenizer = Tokenizer.FromTokenizerJson(modelSet.Tokenizer.bytes);
            speechCodes = metadata.speechTokenIds.Select((id, code) => (id, code)).ToDictionary(x => x.id, x => x.code);
        }
        protected override void OnLoadBackground(OnnxBackend backend)
        {
            try
            {
                // Small autoregressive steps benefit from a bounded thread pool.
                var options = new OnnxSessionOptions { IntraOpThreads = Math.Min(4, Environment.ProcessorCount), InterOpThreads = 1 };
                backbone = backend.CreateSession(modelSet.Backbone, options);
                // The original iSTFT ConvTranspose needs CPU. The equivalent overlap-Conv
                // export can follow the backend policy, including WebGPU.
                codec = backend.CreateSession(modelSet.CodecDecoder, new OnnxSessionOptions
                {
                    IntraOpThreads = options.IntraOpThreads, InterOpThreads = 1,
                    Providers = modelSet.UseCpuCodec ? new[] { OnnxExecutionProvider.Cpu } : Array.Empty<OnnxExecutionProvider>()
                });
                backbone.VerboseLogging = codec.VerboseLogging = VerboseLogging;
            }
            catch { OnUnload(); throw; }
        }
        protected override void OnPrepareInput(TtsRequest request)
        {
            if (request == null || string.IsNullOrWhiteSpace(request.Text)) throw new ArgumentException("Text is required.");
            if (request.VoiceReference != null || !string.IsNullOrWhiteSpace(request.VoiceReferenceText) || !string.IsNullOrWhiteSpace(request.VoiceInstruction))
                throw new ArgumentException("NeuTTS-2E uses its four fixed speakers; custom voice references and instructions are unsupported.");
            if (!string.IsNullOrEmpty(request.LanguageId) && request.LanguageId != "en" && request.LanguageId != "en-us")
                throw new ArgumentException("NeuTTS-2E supports English only.");
            if (request.MaxNewTokens < 1) throw new ArgumentException("MaxNewTokens must be positive.");
        }
        protected override TtsResult OnRun(TtsRequest request, CancellationToken cancellationToken)
        {
            var config = request.NeuTts?.Clone() ?? defaults.Clone();
            long[] ids = NeuTtsPromptBuilder.Build(tokenizer, metadata, request.Text, config);
            if (ids.Length + 50 > metadata.maxContext) throw new ArgumentException("Prompt exceeds the NeuTTS context budget.");
            int budget = Math.Min(request.MaxNewTokens, metadata.maxContext - ids.Length);
            var random = config.UseSeed ? new System.Random(config.Seed) : new System.Random();
            var codes = Generate(ids, budget, config, random, cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            if (codes.Count < 2) throw new InvalidOperationException("NeuTTS generated insufficient speech tokens.");
            using var input = OnnxTensor.FromArray(codes.ToArray(), new[] { 1, 1, codes.Count });
            IReadOnlyDictionary<string, OnnxTensor> output = null;
            try
            {
                output = codec.Run(new Dictionary<string, OnnxTensor> { ["codes"] = input });
                cancellationToken.ThrowIfCancellationRequested();
                return new TtsResult { Samples = (float[])output["audio"].AsFloatArray().Clone(), SampleRate = OutputSampleRate, TokensGenerated = codes.Count };
            }
            finally { Dispose(output); }
        }

        private List<int> Generate(long[] ids, int budget, NeuTtsGenerationConfig config, System.Random random, CancellationToken token)
        {
            var cache = new Dictionary<string, OnnxTensor>();
            var deviceCache = new List<IDeviceTensor>();
            var codes = new List<int>();
            int previous = 0;
            try
            {
                for (int layer = 0; layer < metadata.layers; layer++)
                    foreach (string kind in new[] { "key", "value" })
                        cache.Add($"past_key_values.{layer}.{kind}", OnnxTensor.FromArray(Array.Empty<float>(), new[] { 1, metadata.kvHeads, 0, metadata.headDim }));
                for (int step = 0; step < budget; step++)
                {
                    token.ThrowIfCancellationRequested();
                    int total = previous + ids.Length;
                    var mask = new float[ids.Length * total];
                    for (int row = 0; row < ids.Length; row++)
                        for (int col = previous + row + 1; col < total; col++) mask[row * total + col] = float.NegativeInfinity;
                    using var input = OnnxTensor.FromArray(ids, new[] { 1, ids.Length });
                    using var positions = OnnxTensor.FromArray(Enumerable.Range(previous, ids.Length).Select(x => (long)x).ToArray(), new[] { 1, ids.Length });
                    using var attention = OnnxTensor.FromArray(mask, new[] { 1, 1, ids.Length, total });
                    var inputs = new Dictionary<string, OnnxTensor>(cache) { ["input_ids"] = input, ["position_ids"] = positions, ["attention_mask"] = attention };
                    float[] logits = RunBackbone(inputs, cache, deviceCache);
                    int next = Sample(logits, config, random, step < 50 ? metadata.speechEnd : -1);
                    if (next == metadata.speechEnd) break;
                    if (speechCodes.TryGetValue(next, out int code)) codes.Add(code);
                    previous = total;
                    ids = new[] { (long)next };
                }
                return codes;
            }
            finally
            {
                Dispose(cache);
                foreach (var tensor in deviceCache) tensor.Dispose();
            }
        }

        private float[] RunBackbone(Dictionary<string, OnnxTensor> inputs,
            Dictionary<string, OnnxTensor> cache, List<IDeviceTensor> deviceCache)
        {
            if (backbone is IOnnxDeviceSession deviceSession)
            {
                // Only logits cross back to CPU. Cache tensors remain on the inference
                // device until replaced by the next step, including during prefill.
                var cpuInputs = inputs.Select(pair => new OnnxNamedValue(pair.Key, pair.Value)).ToArray();
                var outputs = deviceSession.RunOnDevice(cpuInputs, deviceCache, new[] { "logits" });
                bool retained = false;
                try
                {
                    using var logits = outputs.Single(tensor => tensor.Name == "logits").ToCpu();
                    foreach (var tensor in deviceCache) tensor.Dispose();
                    deviceCache.Clear();
                    Dispose(cache); cache.Clear();
                    foreach (var tensor in outputs)
                        if (tensor.Name.StartsWith("present.", StringComparison.Ordinal))
                        {
                            tensor.Name = tensor.Name.Replace("present.", "past_key_values.");
                            deviceCache.Add(tensor);
                        }
                    retained = true;
                    return logits.AsFloatArray();
                }
                finally
                {
                    foreach (var tensor in outputs)
                        if (!retained || !tensor.Name.StartsWith("past_key_values.", StringComparison.Ordinal)) tensor.Dispose();
                }
            }

            var cpuOutputs = backbone.Run(inputs);
            try
            {
                Dispose(cache); cache.Clear();
                foreach (var pair in cpuOutputs)
                    if (pair.Key.StartsWith("present.", StringComparison.Ordinal))
                        cache.Add(pair.Key.Replace("present.", "past_key_values."),
                            OnnxTensor.FromArray(pair.Value.AsFloatArray(), pair.Value.Shape));
                return cpuOutputs["logits"].AsFloatArray();
            }
            finally { Dispose(cpuOutputs); }
        }

        internal static int Sample(float[] logits, NeuTtsGenerationConfig config, System.Random random, int suppressed)
        {
            // Keep the full vocabulary, matching upstream sampling and subsequent speech-token extraction.
            var candidates = TopCandidates(logits, config.TopK, suppressed);
            float maximum = logits[candidates[0]];
            double sum = 0;
            var weights = new double[candidates.Length];
            for (int i = 0; i < candidates.Length; i++) sum += weights[i] = Math.Exp((logits[candidates[i]] - maximum) / config.Temperature);
            if (!double.IsFinite(sum) || sum <= 0) throw new InvalidOperationException("NeuTTS produced invalid logits.");
            double cumulative = 0;
            int count = 0;
            while (count < candidates.Length && (count == 0 ||
                (cumulative < config.TopP * sum && weights[count] >= config.MinP * weights[0])))
                cumulative += weights[count++];
            double draw = random.NextDouble() * cumulative;
            for (int i = 0; i < count; i++) if ((draw -= weights[i]) <= 0) return candidates[i];
            return candidates[count - 1];
        }

        internal static int[] TopCandidates(float[] logits, int topK, int suppressed)
        {
            int available = logits.Length - (suppressed >= 0 && suppressed < logits.Length ? 1 : 0);
            if (available == 0) throw new InvalidOperationException("NeuTTS produced no sampling candidates.");
            int size = Math.Min(topK, available);
            // Large user-supplied top-k values favor sorting; the usual top-50 only needs
            // a single vocabulary scan and a small sorted candidate buffer.
            if (size > 256)
                return Enumerable.Range(0, logits.Length).Where(i => i != suppressed)
                    .OrderByDescending(i => logits[i]).Take(size).ToArray();
            var candidates = new int[size];
            int count = 0;
            for (int id = 0; id < logits.Length; id++)
            {
                if (id == suppressed) continue;
                if (count == size && logits[id].CompareTo(logits[candidates[count - 1]]) <= 0) continue;
                int position = Math.Min(count, size - 1);
                while (position > 0 && logits[id].CompareTo(logits[candidates[position - 1]]) > 0)
                {
                    candidates[position] = candidates[position - 1];
                    position--;
                }
                candidates[position] = id;
                if (count < size) count++;
            }
            return candidates;
        }
        private static void Dispose(IReadOnlyDictionary<string, OnnxTensor> tensors)
        { if (tensors != null) foreach (var tensor in tensors.Values) tensor.Dispose(); }
        protected override void OnUnload()
        { backbone?.Dispose(); codec?.Dispose(); backbone = codec = null; tokenizer = null; metadata = null; speechCodes = null; }
    }
}
