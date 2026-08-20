using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using KitsuMate.Tokenizers;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.OmniVoice
{
    public sealed class OmniVoiceEngineRuntime : TtsEngineRuntime
    {
        private readonly OmniVoiceModelSet modelSet;
        private readonly OmniVoiceGenerationConfig defaults;
        private IOnnxSession merged, embeddings, language, heads;
        private IOnnxSession acoustic, semantic, quantizer, decoder;
        private Tokenizer tokenizer;
        private float[] reference24k;
        private float referenceRms;

        public OmniVoiceEngineRuntime(OmniVoiceModelSet set, OmniVoiceGenerationConfig generation, bool verbose)
        {
            modelSet = set;
            defaults = generation?.Clone() ?? new OmniVoiceGenerationConfig();
            VerboseLogging = verbose;
        }

        public override int OutputSampleRate => OmniVoiceConstants.SampleRate;
        public override bool IsMultilingual => true;
        public override string[] SupportedLanguages => Array.Empty<string>();

        public OnnxSessionDiagnostics PrimarySessionDiagnostics =>
            (merged ?? language ?? embeddings)?.Diagnostics;
        public IReadOnlyList<OnnxSessionDiagnostics> SessionDiagnostics => Sessions()
            .Where(session => session != null)
            .Select(session => session.Diagnostics)
            .ToArray();

        protected override void OnLoadMainThread(OnnxBackend backend)
        {
            if (modelSet == null || !modelSet.IsComplete)
                throw new InvalidOperationException("OmniVoiceModelSet is not assigned or incomplete.");
            tokenizer = Tokenizer.FromTokenizerJson(modelSet.Tokenizer.bytes);
        }

        protected override void OnLoadBackground(OnnxBackend backend)
        {
            if (modelSet.Topology == OmniVoiceBackboneTopology.Merged)
            {
                merged = backend.CreateSession(modelSet.MergedBackbone);
                Require(merged, new[] { "input_ids", "audio_mask", "attention_mask", "position_ids" }, new[] { "logits" });
            }
            else
            {
                embeddings = backend.CreateSession(modelSet.AudioEmbeddingsEncoder);
                language = backend.CreateSession(modelSet.LanguageDecoder);
                heads = backend.CreateSession(modelSet.AudioHeadsDecoder);
                Require(embeddings, new[] { "input_ids", "audio_mask" }, new[] { "inputs_embeds" });
                Require(language, new[] { "inputs_embeds", "attention_mask" }, new[] { "hidden_states" });
                Require(heads, new[] { "hidden_states" }, new[] { "logits" });
            }
            acoustic = backend.CreateSession(modelSet.AcousticEncoder);
            semantic = backend.CreateSession(modelSet.SemanticEncoder);
            quantizer = backend.CreateSession(modelSet.QuantizerEncoder);
            decoder = backend.CreateSession(modelSet.HiggsDecoder);
            Require(acoustic, new[] { "waveform_24k" }, new[] { "acoustic_features" });
            Require(semantic, new[] { "waveform_16k" }, new[] { "semantic_features" });
            Require(quantizer, new[] { "acoustic_features", "semantic_features" }, new[] { "codes" });
            Require(decoder, new[] { "codes" }, new[] { "waveform_24k" });
            foreach (IOnnxSession session in Sessions()) if (session != null) session.VerboseLogging = VerboseLogging;
        }

        protected override void OnPrepareInput(TtsRequest request)
        {
            if (request == null) throw new ArgumentNullException(nameof(request));
            if (string.IsNullOrWhiteSpace(request.Text)) throw new ArgumentException("Text is required.", nameof(request));
            OmniVoicePromptBuilder.ResolveMode(request);
            reference24k = null;
            referenceRms = 0f;
            if (request.VoiceReference == null) return;
            AudioClip clip = request.VoiceReference;
            var interleaved = new float[clip.samples * clip.channels];
            if (!clip.GetData(interleaved, 0)) throw new InvalidOperationException("Could not read the reference AudioClip.");
            var mono = new float[clip.samples];
            for (int frame = 0; frame < clip.samples; frame++)
            {
                float sum = 0f;
                for (int channel = 0; channel < clip.channels; channel++) sum += interleaved[frame * clip.channels + channel];
                mono[frame] = sum / clip.channels;
            }
            reference24k = Resample(mono, clip.frequency, OmniVoiceConstants.SampleRate);
            int aligned = reference24k.Length - reference24k.Length % 960;
            if (aligned <= 0) throw new ArgumentException("Reference audio is too short.", nameof(request));
            if (aligned != reference24k.Length) Array.Resize(ref reference24k, aligned);
            referenceRms = Rms(reference24k);
        }

        protected override TtsResult OnRun(TtsRequest request, CancellationToken cancellationToken)
        {
            OmniVoiceGenerationConfig config = request.OmniVoice?.Clone() ?? defaults.Clone();
            config.Validate();
            long[] referenceCodes = reference24k == null ? null : EncodeReference(reference24k);
            int targetFrames = EstimateFrames(request);
            long[] codes = Generate(request, referenceCodes, targetFrames, config, cancellationToken);
            float[] samples = Decode(codes, targetFrames);
            PostProcess(samples, referenceRms);
            return new TtsResult { Samples = samples, SampleRate = OmniVoiceConstants.SampleRate, TokensGenerated = targetFrames };
        }

        private long[] EncodeReference(float[] wave24k)
        {
            float[] wave16k = Resample(wave24k, OmniVoiceConstants.SampleRate, OmniVoiceConstants.SemanticSampleRate);
            OnnxTensor acousticInput = PrecisionTensor(wave24k, new[] { 1, 1, wave24k.Length });
            OnnxTensor semanticInput = PrecisionTensor(wave16k, new[] { 1, wave16k.Length });
            IReadOnlyDictionary<string, OnnxTensor> acousticOutputs = null, semanticOutputs = null, quantizerOutputs = null;
            try
            {
                acousticOutputs = acoustic.Run(new Dictionary<string, OnnxTensor> { ["waveform_24k"] = acousticInput });
                semanticOutputs = semantic.Run(new Dictionary<string, OnnxTensor> { ["waveform_16k"] = semanticInput });
                quantizerOutputs = quantizer.Run(new Dictionary<string, OnnxTensor>
                {
                    ["acoustic_features"] = acousticOutputs["acoustic_features"],
                    ["semantic_features"] = semanticOutputs["semantic_features"]
                });
                return (long[])quantizerOutputs["codes"].AsLongArray().Clone();
            }
            finally
            {
                acousticInput.Dispose(); semanticInput.Dispose();
                Dispose(acousticOutputs); Dispose(semanticOutputs); Dispose(quantizerOutputs);
            }
        }

        private long[] Generate(TtsRequest request, long[] referenceCodes, int targetFrames,
            OmniVoiceGenerationConfig config, CancellationToken cancellationToken)
        {
            long[] style = OmniVoicePromptBuilder.Encode(tokenizer,
                OmniVoicePromptBuilder.StyleText(request.LanguageId, request.VoiceInstruction, config.Denoise, referenceCodes != null));
            long[] text = OmniVoicePromptBuilder.EncodeText(tokenizer,
                OmniVoicePromptBuilder.WrappedText(request.Text, request.VoiceReferenceText));
            int referenceFrames = referenceCodes == null ? 0 : referenceCodes.Length / OmniVoiceConstants.Codebooks;
            int conditionalLength = style.Length + text.Length + referenceFrames + targetFrames;
            int sequence = Math.Max(conditionalLength, targetFrames);
            var ids = Enumerable.Repeat((long)OmniVoiceConstants.AudioMaskId,
                2 * OmniVoiceConstants.Codebooks * sequence).ToArray();
            var audioMask = new bool[2 * sequence];
            var attention = new long[2 * sequence];
            var targets = Enumerable.Repeat((long)OmniVoiceConstants.AudioMaskId,
                OmniVoiceConstants.Codebooks * targetFrames).ToArray();
            int targetStart = conditionalLength - targetFrames;

            for (int codebook = 0; codebook < OmniVoiceConstants.Codebooks; codebook++)
            {
                int conditionalBase = codebook * sequence;
                Array.Copy(style, 0, ids, conditionalBase, style.Length);
                Array.Copy(text, 0, ids, conditionalBase + style.Length, text.Length);
                if (referenceCodes != null)
                    for (int frame = 0; frame < referenceFrames; frame++)
                        ids[conditionalBase + style.Length + text.Length + frame] = referenceCodes[(codebook * referenceFrames) + frame];
            }
            for (int i = 0; i < conditionalLength; i++) attention[i] = 1;
            for (int i = targetStart - referenceFrames; i < conditionalLength; i++) audioMask[i] = true;
            for (int i = 0; i < targetFrames; i++) { attention[sequence + i] = 1; audioMask[sequence + i] = true; }

            int[] schedule = Schedule(targetFrames, config.Steps, config.TimeShift);
            var random = new System.Random(config.Seed);
            for (int step = 0; step < config.Steps; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                float[] logits = RunBackbone(ids, audioMask, attention, sequence);
                var candidates = new List<Candidate>(OmniVoiceConstants.Codebooks * targetFrames);
                for (int codebook = 0; codebook < OmniVoiceConstants.Codebooks; codebook++)
                for (int frame = 0; frame < targetFrames; frame++)
                {
                    int targetIndex = codebook * targetFrames + frame;
                    if (targets[targetIndex] != OmniVoiceConstants.AudioMaskId) continue;
                    int cOffset = (((codebook * sequence) + targetStart + frame) * OmniVoiceConstants.AudioVocabularySize);
                    int uOffset = (((OmniVoiceConstants.Codebooks * sequence) + (codebook * sequence) + frame) * OmniVoiceConstants.AudioVocabularySize);
                    (int token, float score) = Predict(logits, cOffset, uOffset, config, random);
                    score -= codebook * config.LayerPenaltyFactor;
                    if (config.PositionTemperature > 0f) score += Gumbel(random) * config.PositionTemperature;
                    candidates.Add(new Candidate(targetIndex, token, score));
                }
                candidates.Sort((a, b) => b.Score.CompareTo(a.Score));
                int take = Math.Min(schedule[step], candidates.Count);
                for (int i = 0; i < take; i++) targets[candidates[i].Index] = candidates[i].Token;
                for (int codebook = 0; codebook < OmniVoiceConstants.Codebooks; codebook++)
                for (int frame = 0; frame < targetFrames; frame++)
                {
                    long value = targets[codebook * targetFrames + frame];
                    ids[codebook * sequence + targetStart + frame] = value;
                    ids[(OmniVoiceConstants.Codebooks * sequence) + codebook * sequence + frame] = value;
                }
            }
            return targets;
        }

        private float[] RunBackbone(long[] ids, bool[] audioMask, long[] attention, int sequence)
        {
            var inputIds = OnnxTensor.FromArray(ids, new[] { 2, OmniVoiceConstants.Codebooks, sequence });
            var mask = OnnxTensor.FromArray(audioMask, new[] { 2, sequence });
            var attentionTensor = OnnxTensor.FromArray(attention, new[] { 2, sequence });
            try
            {
                if (merged != null)
                {
                    var positions = new long[2 * sequence];
                    for (int batch = 0; batch < 2; batch++) for (int i = 0; i < sequence; i++) positions[batch * sequence + i] = i;
                    using var positionTensor = OnnxTensor.FromArray(positions, new[] { 2, sequence });
                    IReadOnlyDictionary<string, OnnxTensor> outputs = null;
                    try
                    {
                        outputs = merged.Run(new Dictionary<string, OnnxTensor>
                        {
                            ["input_ids"] = inputIds, ["audio_mask"] = mask,
                            ["attention_mask"] = attentionTensor, ["position_ids"] = positionTensor
                        });
                        return ToFloatCopy(outputs["logits"]);
                    }
                    finally { Dispose(outputs); }
                }

                IReadOnlyDictionary<string, OnnxTensor> embedOutputs = null, languageOutputs = null, headOutputs = null;
                try
                {
                    embedOutputs = embeddings.Run(new Dictionary<string, OnnxTensor> { ["input_ids"] = inputIds, ["audio_mask"] = mask });
                    var languageInputs = new Dictionary<string, OnnxTensor>
                    {
                        ["inputs_embeds"] = embedOutputs["inputs_embeds"], ["attention_mask"] = attentionTensor
                    };
                    foreach (string name in language.InputNames.Where(n => n.StartsWith("past_key_values.", StringComparison.Ordinal)))
                        languageInputs[name] = OnnxTensor.FromArray(Array.Empty<float>(), new[] { 2, 8, 0, 128 });
                    languageOutputs = language.Run(languageInputs);
                    foreach (var pair in languageInputs.Where(p => p.Key.StartsWith("past_key_values.", StringComparison.Ordinal))) pair.Value.Dispose();
                    headOutputs = heads.Run(new Dictionary<string, OnnxTensor> { ["hidden_states"] = languageOutputs["hidden_states"] });
                    return ToFloatCopy(headOutputs["logits"]);
                }
                finally { Dispose(embedOutputs); Dispose(languageOutputs); Dispose(headOutputs); }
            }
            finally { inputIds.Dispose(); mask.Dispose(); attentionTensor.Dispose(); }
        }

        private float[] Decode(long[] codes, int frames)
        {
            using var input = OnnxTensor.FromArray(codes, new[] { OmniVoiceConstants.Codebooks, 1, frames });
            IReadOnlyDictionary<string, OnnxTensor> outputs = null;
            try { outputs = decoder.Run(new Dictionary<string, OnnxTensor> { ["codes"] = input }); return ToFloatCopy(outputs["waveform_24k"]); }
            finally { Dispose(outputs); }
        }

        private OnnxTensor PrecisionTensor(float[] values, int[] shape) => modelSet.CodecPrecision == OmniVoiceTensorPrecision.Float16
            ? OnnxTensor.FromArray(values.Select(FloatToHalf).ToArray(), shape)
            : OnnxTensor.FromArray(values, shape);

        internal static (int Token, float Score) Predict(float[] logits, int conditionalOffset, int unconditionalOffset,
            OmniVoiceGenerationConfig config, System.Random random)
        {
            float cMax = float.NegativeInfinity, uMax = float.NegativeInfinity;
            for (int token = 0; token < OmniVoiceConstants.AudioVocabularySize; token++)
            { cMax = Math.Max(cMax, logits[conditionalOffset + token]); uMax = Math.Max(uMax, logits[unconditionalOffset + token]); }
            double cSum = 0, uSum = 0;
            for (int token = 0; token < OmniVoiceConstants.AudioVocabularySize; token++)
            { cSum += Math.Exp(logits[conditionalOffset + token] - cMax); uSum += Math.Exp(logits[unconditionalOffset + token] - uMax); }
            float cNorm = cMax + (float)Math.Log(cSum), uNorm = uMax + (float)Math.Log(uSum);
            int best = 0; float bestScore = float.NegativeInfinity;
            for (int token = 0; token < OmniVoiceConstants.AudioMaskId; token++)
            {
                float c = logits[conditionalOffset + token] - cNorm;
                float u = logits[unconditionalOffset + token] - uNorm;
                float score = c + config.GuidanceScale * (c - u);
                if (config.ClassTemperature > 0f) score = score / config.ClassTemperature + Gumbel(random);
                if (score > bestScore) { bestScore = score; best = token; }
            }
            return (best, bestScore);
        }

        internal static int[] Schedule(int frames, int steps, float shift)
        {
            int total = frames * OmniVoiceConstants.Codebooks, remaining = total;
            var result = new int[steps];
            float previous = 0f;
            for (int step = 0; step < steps; step++)
            {
                float t = (step + 1f) / steps;
                float shifted = shift * t / (1f + (shift - 1f) * t);
                int count = step == steps - 1 ? remaining : Math.Min(remaining, Mathf.CeilToInt(total * (shifted - previous)));
                result[step] = Math.Max(0, count); remaining -= result[step]; previous = shifted;
            }
            return result;
        }

        internal static int EstimateFrames(TtsRequest request)
        {
            if (request.DurationSeconds > 0f) return Math.Max(1, Mathf.RoundToInt(request.DurationSeconds * OmniVoiceConstants.FrameRate));
            float speed = request.Speed > 0f ? request.Speed : 1f;
            float seconds = Mathf.Clamp(request.Text.Trim().Length / 12f, 0.8f, 30f) / speed;
            return Math.Max(1, Mathf.RoundToInt(seconds * OmniVoiceConstants.FrameRate));
        }

        private static float Gumbel(System.Random random)
        {
            double u = Math.Max(1e-10, Math.Min(1d - 1e-10, random.NextDouble()));
            return (float)(-Math.Log(-Math.Log(u)));
        }

        private static void PostProcess(float[] samples, float targetRms)
        {
            float rms = Rms(samples);
            float scale = targetRms > 0f && rms > 1e-6f ? Math.Min(2f, targetRms / rms) : 1f;
            int fade = Math.Min(samples.Length / 2, OmniVoiceConstants.SampleRate / 20);
            for (int i = 0; i < samples.Length; i++)
            {
                float value = float.IsFinite(samples[i]) ? samples[i] * scale : 0f;
                if (i < fade) value *= i / (float)fade;
                if (i >= samples.Length - fade) value *= (samples.Length - 1 - i) / (float)fade;
                samples[i] = Mathf.Clamp(value, -1f, 1f);
            }
        }

        private static float Rms(float[] samples)
        {
            if (samples == null || samples.Length == 0) return 0f;
            double sum = 0; foreach (float value in samples) sum += value * value;
            return (float)Math.Sqrt(sum / samples.Length);
        }

        private static float[] Resample(float[] source, int sourceRate, int targetRate)
        {
            if (sourceRate == targetRate) return (float[])source.Clone();
            int length = Math.Max(1, (int)Math.Round(source.Length * (double)targetRate / sourceRate));
            var result = new float[length]; double ratio = (double)sourceRate / targetRate;
            for (int i = 0; i < length; i++)
            {
                double at = i * ratio; int left = Math.Min((int)at, source.Length - 1), right = Math.Min(left + 1, source.Length - 1);
                result[i] = Mathf.Lerp(source[left], source[right], (float)(at - left));
            }
            return result;
        }

        private static float[] ToFloatCopy(OnnxTensor tensor)
        {
            if (tensor.ElementType == OnnxTensorElementType.Float) return (float[])tensor.AsFloatArray().Clone();
            if (tensor.ElementType == OnnxTensorElementType.Float16) return tensor.AsFloat16Array().Select(HalfToFloat).ToArray();
            throw new InvalidOperationException($"Expected floating-point tensor, got {tensor.ElementType}.");
        }

        private static ushort FloatToHalf(float value)
        {
            uint bits = BitConverter.ToUInt32(BitConverter.GetBytes(value), 0);
            uint sign = (bits >> 16) & 0x8000, exponent = (bits >> 23) & 0xff, mantissa = bits & 0x7fffff;
            if (exponent == 255) return (ushort)(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
            int halfExponent = (int)exponent - 127 + 15;
            if (halfExponent >= 31) return (ushort)(sign | 0x7c00);
            if (halfExponent <= 0)
            {
                if (halfExponent < -10) return (ushort)sign;
                mantissa = (mantissa | 0x800000) >> (1 - halfExponent);
                return (ushort)(sign | ((mantissa + 0x1000) >> 13));
            }
            return (ushort)(sign | ((uint)halfExponent << 10) | ((mantissa + 0x1000) >> 13));
        }

        private static float HalfToFloat(ushort value)
        {
            uint sign = (uint)(value & 0x8000) << 16, exponent = (uint)(value >> 10) & 0x1f, mantissa = (uint)value & 0x3ff;
            uint bits;
            if (exponent == 0)
            {
                if (mantissa == 0) bits = sign;
                else { exponent = 1; while ((mantissa & 0x400) == 0) { mantissa <<= 1; exponent--; } mantissa &= 0x3ff; bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13); }
            }
            else if (exponent == 31) bits = sign | 0x7f800000 | (mantissa << 13);
            else bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
            return BitConverter.ToSingle(BitConverter.GetBytes(bits), 0);
        }

        private static void Require(IOnnxSession session, IEnumerable<string> inputs, IEnumerable<string> outputs)
        {
            foreach (string input in inputs) if (!session.InputNames.Contains(input)) throw new OnnxModelContractException($"Model is missing input '{input}'.");
            foreach (string output in outputs) if (!session.OutputNames.Contains(output)) throw new OnnxModelContractException($"Model is missing output '{output}'.");
        }

        private IEnumerable<IOnnxSession> Sessions()
        { yield return merged; yield return embeddings; yield return language; yield return heads; yield return acoustic; yield return semantic; yield return quantizer; yield return decoder; }

        protected override void OnUnload()
        {
            foreach (IOnnxSession session in Sessions()) session?.Dispose();
            merged = embeddings = language = heads = acoustic = semantic = quantizer = decoder = null;
            tokenizer = null; reference24k = null;
        }

        private static void Dispose(IReadOnlyDictionary<string, OnnxTensor> values)
        { if (values == null) return; foreach (OnnxTensor value in values.Values) value?.Dispose(); }

        private readonly struct Candidate
        {
            public Candidate(int index, int token, float score) { Index = index; Token = token; Score = score; }
            public int Index { get; } public int Token { get; } public float Score { get; }
        }
    }
}
