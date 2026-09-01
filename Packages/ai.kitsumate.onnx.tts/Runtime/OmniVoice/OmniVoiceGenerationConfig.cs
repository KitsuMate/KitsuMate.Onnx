using System;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.OmniVoice
{
    [Serializable]
    public sealed class OmniVoiceGenerationConfig
    {
        [Min(1)] public int Steps = 32;
        [Min(0f)] public float GuidanceScale = 2f;
        [Min(0.0001f)] public float TimeShift = 0.1f;
        [Min(0f)] public float LayerPenaltyFactor = 5f;
        [Min(0f)] public float PositionTemperature = 5f;
        [Min(0f)] public float ClassTemperature;
        public bool Denoise = true;
        public int Seed = 0;

        public OmniVoiceGenerationConfig Clone() => (OmniVoiceGenerationConfig)MemberwiseClone();

        internal void Validate()
        {
            if (Steps < 1) throw new ArgumentOutOfRangeException(nameof(Steps));
            if (GuidanceScale < 0f) throw new ArgumentOutOfRangeException(nameof(GuidanceScale));
            if (TimeShift <= 0f) throw new ArgumentOutOfRangeException(nameof(TimeShift));
            if (LayerPenaltyFactor < 0f || PositionTemperature < 0f || ClassTemperature < 0f)
                throw new ArgumentOutOfRangeException("OmniVoice temperatures and penalties must be non-negative.");
        }
    }
}
