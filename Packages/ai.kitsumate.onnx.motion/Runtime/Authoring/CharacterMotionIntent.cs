using System;
using UnityEngine;
using KitsuMate.Onnx.Embeddings;

namespace KitsuMate.Onnx.Motion
{
    [Serializable]
    public struct CharacterMotionGenerationSettings
    {
        [Min(2)] public int DenoisingSteps;
        public int Seed;
        public float TextGuidance;
        public float ConstraintGuidance;
        public float FirstHeadingRadians;

        public static CharacterMotionGenerationSettings Default => new CharacterMotionGenerationSettings
        {
            DenoisingSteps = 25,
            Seed = 0,
            TextGuidance = 2f,
            ConstraintGuidance = 2f,
            FirstHeadingRadians = 0f,
        };

        public CharacterMotionGenerationSettings WithDefaults()
        {
            CharacterMotionGenerationSettings result = this;
            if (result.DenoisingSteps < 2) result.DenoisingSteps = Default.DenoisingSteps;
            return result;
        }

        public KimodoGenerationRequest CreateRequest(KimodoConstraintSet constraints) => new KimodoGenerationRequest(
            Seed,
            DenoisingSteps < 2 ? 25 : DenoisingSteps,
            TextGuidance,
            ConstraintGuidance,
            FirstHeadingRadians,
            KimodoConditioning.DefaultFrameCount,
            constraints);
    }

    public interface ICharacterMotionIntent
    {
        string Prompt { get; }
        CharacterMotionGenerationSettings Settings { get; }
        bool TryGetEmbedding(ModelIdentity encoder, out KimodoTextEmbedding embedding);
    }

    [CreateAssetMenu(fileName = "CharacterMotionIntent", menuName = "KitsuMate/Motion/Character Motion Intent")]
    public sealed class CharacterMotionIntent : ScriptableObject, ICharacterMotionIntent
    {
        [SerializeField] private string id;
        [SerializeField, TextArea(2, 5)] private string description;
        [SerializeField, TextArea(2, 6)] private string prompt;
        [SerializeField] private bool useCustomGenerationSettings = true;
        [SerializeField] private CharacterMotionGenerationSettings settings = default;
        [Header("Embedding")]
        [SerializeField] private EmbeddingEngine embeddingEngine;

        [Header("Optional baked embedding")]
        [SerializeField, HideInInspector] private float[] embedding;
        [SerializeField, HideInInspector] private string embeddingModelIdentity;
        [SerializeField, HideInInspector] private string embeddingPromptHash;

        public string Id => id;
        public string Description => description;
        public string Prompt => prompt ?? string.Empty;
        public bool UsesCustomGenerationSettings => useCustomGenerationSettings;
        public CharacterMotionGenerationSettings Settings => useCustomGenerationSettings
            ? settings.WithDefaults()
            : CharacterMotionGenerationSettings.Default;
        public bool HasEmbedding => embedding != null && embedding.Length == KimodoTextEmbedding.Dimension;
        public EmbeddingEngine EmbeddingEngine => embeddingEngine;
        public string PromptHash => ComputePromptHash(Prompt);

        public bool TryGetEmbedding(ModelIdentity encoder, out KimodoTextEmbedding result)
        {
            result = null;
            if (!HasEmbedding || !string.Equals(embeddingPromptHash, PromptHash, StringComparison.Ordinal)) return false;
            if (!string.Equals(embeddingModelIdentity ?? string.Empty, encoder.ToString(), StringComparison.Ordinal)) return false;
            result = new KimodoTextEmbedding(embedding, encoder.ToString());
            return true;
        }

        public void SetEmbedding(ModelIdentity encoder, KimodoTextEmbedding value)
        {
            if (value == null) throw new ArgumentNullException(nameof(value));
            embedding = value.Values.ToArray();
            embeddingModelIdentity = encoder.ToString();
            embeddingPromptHash = PromptHash;
        }

        public void ClearEmbedding()
        {
            embedding = null;
            embeddingModelIdentity = embeddingPromptHash = null;
        }

        internal static string ComputePromptHash(string value) => Hash128.Compute(value ?? string.Empty).ToString();

        private void Reset()
        {
            useCustomGenerationSettings = false;
            settings = CharacterMotionGenerationSettings.Default;
        }
    }
}
