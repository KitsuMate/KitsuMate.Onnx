using System.Linq;
using UnityEngine;
using UnityEngine.Serialization;
using KitsuMate.Onnx;

namespace KitsuMate.Onnx.Asr.Whisper
{
    /// <summary>
    /// Groups Whisper ONNX models together for easy variant swapping.
    /// Contains encoder, decoder models. Mel spectrogram processor is bundled in the package.
    /// </summary>
    [CreateAssetMenu(fileName = "WhisperModelSet", menuName = "KitsuMate/ONNX/ASR/Whisper Model Set")]
    public class WhisperModelSet : StandardModelSet
    {
        [SerializeField] private OnnxModelReference _melProcessorSource = new();
        [SerializeField] private OnnxModelReference _encoderSource = new();
        [SerializeField] private OnnxModelReference _decoderSource = new();

        [Header("Tokenizer")]
        [FormerlySerializedAs("_vocabulary")]
        [SerializeField, Tooltip("Tokenizer JSON file with the full Whisper tokenization pipeline")]
        private TextAsset _tokenizerJson;
        
        
        /// <summary>Mel spectrogram processor model.</summary>
        public OnnxModelReference MelProcessor => _melProcessorSource;
        
        /// <summary>Audio encoder model.</summary>
        public OnnxModelReference Encoder => _encoderSource;
        
        /// <summary>Text decoder model.</summary>
        public OnnxModelReference Decoder => _decoderSource;
        
        /// <summary>Tokenizer JSON file.</summary>
        public TextAsset TokenizerJson => _tokenizerJson;
        
        public override string DisplayName => string.IsNullOrWhiteSpace(name) ? "Whisper" : name;
        
        public override bool IsComplete => 
            _melProcessorSource.IsAvailable &&
            _encoderSource.IsAvailable &&
            _decoderSource.IsAvailable &&
            _tokenizerJson != null;
        
        public override IOnnxModelSource[] GetAllModels()
        {
            return new IOnnxModelSource[] { _melProcessorSource, _encoderSource, _decoderSource };
        }
        
        public override ModelValidationResult Validate(ModelValidationContext context)
        {
            var result = new ModelValidationResult();
            if (!_melProcessorSource.IsAvailable) result.Error("missing_mel", "Mel processor model is not available.");
            if (!_encoderSource.IsAvailable) result.Error("missing_encoder", "Encoder model is not available.");
            if (!_decoderSource.IsAvailable) result.Error("missing_decoder", "Decoder model is not available.");
            if (_tokenizerJson == null) result.Error("missing_tokenizer", "Tokenizer JSON file is not assigned.");
            RequireInput(_melProcessorSource, "audio", result);
            if (_encoderSource.HasInspectedMetadata &&
                !_encoderSource.Inputs.Any(input => input.Name == "mel" || input.Name == "input_features"))
                result.Error("missing_input", $"Model '{_encoderSource.SourceName}' is missing input 'input_features'.");
            RequireInput(_decoderSource, "encoder_hidden_states", result);
            RequireInput(_decoderSource, "input_ids", result);
            foreach (IOnnxModelSource model in GetAllModels()) RequireSchema(model, result);
            return ValidateCommon(context, result);
        }
        
#if UNITY_EDITOR
        public void SetTokenizer(TextAsset tokenizerJson)
        {
            _tokenizerJson = tokenizerJson;
            UnityEditor.EditorUtility.SetDirty(this);
        }

        /// <summary>
        /// Sets the models programmatically. Editor-only.
        /// </summary>
        public void SetModels(OnnxModelAsset melProcessor, OnnxModelAsset encoder, OnnxModelAsset decoder, TextAsset tokenizerJson, string modelIdentifier = null)
        {
            _melProcessorSource.ConfigureAsset(melProcessor);
            _encoderSource.ConfigureAsset(encoder);
            _decoderSource.ConfigureAsset(decoder);
            _tokenizerJson = tokenizerJson;
            UnityEditor.EditorUtility.SetDirty(this);
        }
#endif
    }
}
