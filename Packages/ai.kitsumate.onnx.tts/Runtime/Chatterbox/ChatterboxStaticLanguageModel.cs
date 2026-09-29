using System;
using System.Collections.Generic;
using System.Linq;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    /// <summary>
    /// Runs the split V3 language model over a fixed-size KV cache that stays on the inference device.
    /// Token embeddings run on the CPU; the transformer reads them, writes the new cache rows in place,
    /// and returns final-position logits. Because every decode step binds the same tensors with the
    /// same shapes, providers with graph capture record one step and replay it for every later token.
    /// </summary>
    /// <remarks>
    /// When the provider fails, the session falls back to the next one and the tensors of the failed
    /// provider are created again on the new one. Setup and prefill then simply repeat; a failed decode
    /// step has lost its cache, so <see cref="Step"/> rethrows and the caller starts the request again.
    /// </remarks>
    internal sealed class ChatterboxStaticLanguageModel : IDisposable
    {
        /// <summary>Rows are the text-conditioned and unconditioned guidance pair; guidance 0 reads only the first.</summary>
        public const int Batch = 2;

        private const int ConditioningWidth = 1024;
        private const int DecodeGraph = 0;

        private readonly IOnnxSession embedding;
        private readonly IOnnxDeviceSession languageModel;
        private readonly string[] caches;
        private readonly List<IDeviceTensor> owned = new();
        private readonly Dictionary<string, IDeviceTensor> inputs = new(StringComparer.Ordinal);
        private readonly Dictionary<string, IDeviceTensor> outputs = new(StringComparer.Ordinal);
        private IDeviceTensor stepEmbeds, sequenceLengths, totalLength, logits;
        private int length;

        public ChatterboxStaticLanguageModel(IOnnxSession embedding, IOnnxSession languageModel, int capacity)
        {
            this.embedding = embedding ?? throw new ArgumentNullException(nameof(embedding));
            this.languageModel = languageModel as IOnnxDeviceSession ?? throw new OnnxModelContractException(
                "The Chatterbox split language model needs a backend with device-resident tensors.");
            Capacity = capacity;
            caches = languageModel.OutputNames.Where(name => name.StartsWith("present.", StringComparison.Ordinal))
                .ToArray();
            if (caches.Length == 0 || caches.Any(name => !languageModel.InputNames.Contains(PastName(name))))
                throw new OnnxModelContractException("The split language model has an incomplete KV cache contract.");
            try { Prepare(); }
            catch { Dispose(); throw; }
        }

        /// <summary>Most positions (conditioning, text, and generated tokens) one request can hold.</summary>
        public int Capacity { get; }

        /// <summary>
        /// Starts a request: embeds conditioning plus text and fills the cache from position 0.
        /// Returns logits for both guidance rows.
        /// </summary>
        public float[] Prefill(long[] ids, long[] positions, OnnxTensor conditioning, float exaggeration)
        {
            using OnnxTensor embeds = Embed(ids, positions, conditioning, exaggeration);
            int count = embeds.Shape[1];
            if (count > Capacity)
                throw new ArgumentException($"The request needs {count} language model positions; at most {Capacity} fit.");
            while (true)
            {
                try
                {
                    length = count;
                    SetLength(count);
                    var prefillInputs = new Dictionary<string, IDeviceTensor>(inputs, StringComparer.Ordinal);
                    prefillInputs.Remove("inputs_embeds");
                    languageModel.RunBound(new[] { new OnnxNamedValue("inputs_embeds", embeds) }, prefillInputs, outputs);
                    return ReadLogits();
                }
                // Prefill writes the cache from position 0, so it can simply run again on the new provider.
                catch (OnnxProviderFallbackException) { Prepare(); }
            }
        }

        /// <summary>Appends one generated speech token and returns the next logits for both guidance rows.</summary>
        /// <exception cref="OnnxProviderFallbackException">
        /// The provider failed and the cache was lost. The model is ready again on the new provider;
        /// start the request again with <see cref="Prefill"/>.
        /// </exception>
        public float[] Step(long token, long position, float exaggeration)
        {
            if (length >= Capacity) throw new InvalidOperationException("The language model cache is full.");
            using var conditioning = OnnxTensor.FromArray(Array.Empty<float>(), new[] { Batch, 0, ConditioningWidth });
            try
            {
                using (OnnxTensor embeds = Embed(new[] { token }, new[] { position }, conditioning, exaggeration))
                    stepEmbeds.CopyFrom(embeds);
                SetLength(++length);
                languageModel.RunBound(null, inputs, outputs, DecodeGraph);
                return ReadLogits();
            }
            catch (OnnxProviderFallbackException)
            {
                Prepare();
                throw;
            }
        }

        /// <summary>
        /// Creates the fixed tensors and records the decode step. A provider fallback during either
        /// leaves tensors of the failed provider, so they are released and created on the new one.
        /// Each fallback moves to a later provider, and the session throws when none remains.
        /// </summary>
        private void Prepare()
        {
            while (true)
            {
                ReleaseTensors();
                try
                {
                    CreateTensors();
                    CaptureDecodeStep();
                    return;
                }
                catch (OnnxProviderFallbackException) { }
            }
        }

        private void CreateTensors()
        {
            stepEmbeds = Create("inputs_embeds", OnnxTensorElementType.Float, new[] { Batch, 1, ConditioningWidth });
            sequenceLengths = Create("seqlens_k", OnnxTensorElementType.Int32, new[] { Batch });
            totalLength = Create("total_sequence_length", OnnxTensorElementType.Int32, Array.Empty<int>());
            logits = Create("logits", OnnxTensorElementType.Float, new[] { Batch, ChatterboxConstants.SpeechLogitCount });
            inputs["inputs_embeds"] = stepEmbeds;
            inputs["seqlens_k"] = sequenceLengths;
            inputs["total_sequence_length"] = totalLength;
            outputs["logits"] = logits;
            int[] cacheShape = { Batch, ChatterboxConstants.NumKeyValueHeads, Capacity, ChatterboxConstants.HeadDim };
            using var zeros = OnnxTensor.FromArray(new float[cacheShape.Aggregate(1, (a, b) => a * b)], cacheShape);
            foreach (string present in caches)
            {
                // Rows past the current length are masked, but they must not hold NaN from a reused buffer.
                // GroupQueryAttention supports one buffer as both past and present cache; see RunBound.
                IDeviceTensor cache = Create(present, OnnxTensorElementType.Float, cacheShape);
                cache.CopyFrom(zeros);
                inputs[PastName(present)] = cache;
                outputs[present] = cache;
            }
        }

        /// <summary>
        /// Records the decode graph while the cache is full, so every length-dependent dispatch size
        /// covers the whole buffer. Later steps only mask by length. Recording at a shorter length
        /// leaves attention tiles past that length unprocessed on the WebGPU provider.
        /// </summary>
        private void CaptureDecodeStep()
        {
            using var zeros = OnnxTensor.FromArray(new float[Batch * ConditioningWidth], new[] { Batch, 1, ConditioningWidth });
            stepEmbeds.CopyFrom(zeros);
            SetLength(Capacity);
            languageModel.RunBound(null, inputs, outputs, DecodeGraph);
            length = 0;
        }

        private OnnxTensor Embed(long[] ids, long[] positions, OnnxTensor conditioning, float exaggeration)
        {
            using var idTensor = OnnxTensor.FromArray(Repeat(ids), new[] { Batch, ids.Length });
            using var positionTensor = OnnxTensor.FromArray(Repeat(positions), new[] { Batch, positions.Length });
            using var exaggerationTensor = OnnxTensor.FromArray(new[] { exaggeration, exaggeration }, new[] { Batch });
            using var textConditioning = OnnxTensor.FromArray(new[] { 1f, 0f }, new[] { Batch });
            var result = embedding.Run(new Dictionary<string, OnnxTensor>
            {
                ["input_ids"] = idTensor, ["token_position_ids"] = positionTensor, ["exaggeration"] = exaggerationTensor,
                ["text_conditioning"] = textConditioning, ["conditioning"] = conditioning
            });
            foreach (var pair in result) if (pair.Key != "inputs_embeds") pair.Value.Dispose();
            return result["inputs_embeds"];
        }

        private void SetLength(int value)
        {
            using var lengths = OnnxTensor.FromArray(new[] { value - 1, value - 1 }, new[] { Batch });
            using var total = OnnxTensor.FromArray(new[] { value }, Array.Empty<int>());
            sequenceLengths.CopyFrom(lengths);
            totalLength.CopyFrom(total);
        }

        private float[] ReadLogits()
        {
            using OnnxTensor values = logits.ToCpu();
            return values.AsFloatArray();
        }

        private IDeviceTensor Create(string name, OnnxTensorElementType type, int[] shape)
        {
            IDeviceTensor tensor = languageModel.CreateDeviceTensor(name, type, shape);
            owned.Add(tensor);
            return tensor;
        }

        private void ReleaseTensors()
        {
            foreach (IDeviceTensor tensor in owned) tensor.Dispose();
            owned.Clear();
            inputs.Clear();
            outputs.Clear();
            stepEmbeds = sequenceLengths = totalLength = logits = null;
        }

        private static long[] Repeat(long[] values)
        {
            var result = new long[values.Length * Batch];
            for (int row = 0; row < Batch; row++) Array.Copy(values, 0, result, row * values.Length, values.Length);
            return result;
        }

        private static string PastName(string present) => "past_key_values." + present.Substring("present.".Length);

        public void Dispose() => ReleaseTensors();
    }
}
