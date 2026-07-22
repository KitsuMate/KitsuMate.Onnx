using System;

namespace KitsuMate.Onnx.Motion
{
    /// <summary>One pooled LLM2Vec embedding used by Kimodo.</summary>
    public sealed class KimodoTextEmbedding
    {
        public const int Dimension = 4096;

        private readonly float[] _values;

        public ReadOnlyMemory<float> Values => _values;
        public string ModelId { get; }

        public KimodoTextEmbedding(float[] values, string modelId = null, bool copy = true)
        {
            if (values == null) throw new ArgumentNullException(nameof(values));
            if (values.Length != Dimension)
                throw new ArgumentException($"Kimodo embeddings must contain {Dimension} values.", nameof(values));

            for (int i = 0; i < values.Length; i++)
            {
                if (float.IsNaN(values[i]) || float.IsInfinity(values[i]))
                    throw new ArgumentException($"Embedding contains a non-finite value at index {i}.", nameof(values));
            }

            _values = copy ? (float[])values.Clone() : values;
            ModelId = modelId ?? string.Empty;
        }
    }
}
