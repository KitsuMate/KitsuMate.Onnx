using System;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    [Serializable]
    public sealed class ChatterboxGenerationConfig
    {
        public float Temperature = 0.8f;
        public float TopP = 0.95f;
        public float MinP = 0.05f;
        public float Guidance = 0.5f;
        public int Seed = 42;

        public void Validate()
        {
            if (!float.IsFinite(Temperature) || Temperature < 0 ||
                !float.IsFinite(Guidance) || Guidance < 0 ||
                !float.IsFinite(TopP) || TopP <= 0 || TopP > 1 ||
                !float.IsFinite(MinP) || MinP < 0 || MinP > 1)
                throw new ArgumentException("Chatterbox requires finite nonnegative temperature and guidance, top-p in (0, 1], and min-p in [0, 1].");
        }
    }
}
