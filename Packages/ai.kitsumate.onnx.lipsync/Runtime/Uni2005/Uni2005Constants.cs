namespace KitsuMate.Onnx.LipSync.Uni2005
{
    /// <summary>
    /// Configuration constants for Uni2005 phoneme detection.
    /// </summary>
    public static class Uni2005Constants
    {
        /// <summary>Sample rate expected by the model (8kHz).</summary>
        public const int SampleRate = 8000;
        
        /// <summary>Frame length in seconds for MFCC extraction.</summary>
        public const float FrameLengthSeconds = 0.025f;
        
        /// <summary>Frame step/hop in seconds.</summary>
        public const float FrameStepSeconds = 0.01f;
        
        /// <summary>Number of MFCC coefficients.</summary>
        public const int CepstralCoefficients = 40;
        
        /// <summary>Number of mel filter banks.</summary>
        public const int MelBanks = 40;
        
        /// <summary>Low frequency cutoff for mel filterbank.</summary>
        public const float LowFrequency = 40f;
        
        /// <summary>High frequency cutoff for mel filterbank.</summary>
        public const float HighFrequency = 3800f;
        
        /// <summary>Pre-emphasis coefficient.</summary>
        public const float PreEmphasis = 0.97f;
        
        /// <summary>Cepstral liftering coefficient.</summary>
        public const int CeplifterCoeff = 22;
        
        /// <summary>Default batch size for inference.</summary>
        public const int DefaultBatchSize = 512;
        
        /// <summary>Default volume threshold for silence detection.</summary>
        public const float DefaultVolumeThreshold = 0.01f;
    }
}
