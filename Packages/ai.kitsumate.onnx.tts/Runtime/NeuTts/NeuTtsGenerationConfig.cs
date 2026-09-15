using System;

namespace KitsuMate.Onnx.Tts.NeuTts
{
    [Serializable]
    public sealed class NeuTtsGenerationConfig
    {
        public string Speaker = "emily";
        public string Emotion = "neutral";
        public float Temperature = 1f;
        public int TopK = 50;
        public float TopP = 1f;
        public float MinP = 0f;
        public bool UseSeed;
        public int Seed;

        public NeuTtsGenerationConfig Clone() => (NeuTtsGenerationConfig)MemberwiseClone();
        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(Speaker) || string.IsNullOrWhiteSpace(Emotion))
                throw new ArgumentException("Speaker and emotion are required.");
            if (!float.IsFinite(Temperature) || Temperature <= 0f || TopK < 1)
                throw new ArgumentException("Temperature and top-k must be positive.");
            if (!float.IsFinite(TopP) || TopP <= 0f || TopP > 1f ||
                !float.IsFinite(MinP) || MinP < 0f || MinP > 1f)
                throw new ArgumentException("Top-p must be in (0, 1] and min-p in [0, 1].");
        }
    }
}
