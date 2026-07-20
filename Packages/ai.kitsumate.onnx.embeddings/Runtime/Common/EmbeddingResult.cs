using System;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings
{
    /// <summary>
    /// Result containing embedding vectors.
    /// </summary>
    [Serializable]
    public class EmbeddingResult
    {
        [SerializeField] private float[] _embedding;
        [SerializeField] private int _dimension;
        [SerializeField] private float _computeTimeMs;
        [SerializeField] private bool _normalized;
        [SerializeField] private string _modelIdentity;
        
        /// <summary>The embedding vector.</summary>
        public float[] Embedding => _embedding;
        
        /// <summary>Dimension of the embedding.</summary>
        public int Dimension => _dimension;
        
        /// <summary>Time taken to compute in milliseconds.</summary>
        public float ComputeTimeMs => _computeTimeMs;
        public bool IsNormalized => _normalized;
        public string ModelIdentity => _modelIdentity;
        public float[][] Batch { get; internal set; }
        
        public EmbeddingResult(float[] embedding, float computeTimeMs = 0, bool normalized = false, ModelIdentity modelIdentity = default)
        {
            _embedding = embedding;
            _dimension = embedding?.Length ?? 0;
            _computeTimeMs = computeTimeMs;
            _normalized = normalized;
            _modelIdentity = modelIdentity.ToString();
        }
        
        /// <summary>
        /// Compute cosine similarity with another embedding.
        /// </summary>
        public float CosineSimilarity(EmbeddingResult other)
        {
            return CosineSimilarity(other?.Embedding);
        }
        
        /// <summary>
        /// Compute cosine similarity with another embedding vector.
        /// </summary>
        public float CosineSimilarity(float[] other)
        {
            if (_embedding == null || other == null || _embedding.Length != other.Length)
                return 0f;
            
            float dot = 0f, normA = 0f, normB = 0f;
            for (int i = 0; i < _embedding.Length; i++)
            {
                dot += _embedding[i] * other[i];
                normA += _embedding[i] * _embedding[i];
                normB += other[i] * other[i];
            }
            
            if (normA <= 0 || normB <= 0)
                return 0f;
            
            return dot / (Mathf.Sqrt(normA) * Mathf.Sqrt(normB));
        }
        
        /// <summary>
        /// Compute euclidean distance with another embedding.
        /// </summary>
        public float EuclideanDistance(EmbeddingResult other)
        {
            return EuclideanDistance(other?.Embedding);
        }
        
        /// <summary>
        /// Compute euclidean distance with another embedding vector.
        /// </summary>
        public float EuclideanDistance(float[] other)
        {
            if (_embedding == null || other == null || _embedding.Length != other.Length)
                return float.MaxValue;
            
            float sum = 0f;
            for (int i = 0; i < _embedding.Length; i++)
            {
                float diff = _embedding[i] - other[i];
                sum += diff * diff;
            }
            
            return Mathf.Sqrt(sum);
        }
        
        /// <summary>
        /// Normalize the embedding to unit length.
        /// </summary>
        public void Normalize()
        {
            if (_embedding == null)
                return;
            
            float norm = 0f;
            for (int i = 0; i < _embedding.Length; i++)
                norm += _embedding[i] * _embedding[i];
            
            if (norm > 0)
            {
                norm = Mathf.Sqrt(norm);
                for (int i = 0; i < _embedding.Length; i++)
                    _embedding[i] /= norm;
            }
        }
    }
}
