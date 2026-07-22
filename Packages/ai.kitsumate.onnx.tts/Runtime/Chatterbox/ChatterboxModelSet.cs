using UnityEngine;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    [CreateAssetMenu(fileName = "ChatterboxModelSet", menuName = "KitsuMate/ONNX/TTS/Chatterbox Model Set")]
    public class ChatterboxModelSet : StandardModelSet
    {
        [SerializeField] private OnnxModelReference _speechEncoderSource = new();
        [SerializeField] private OnnxModelReference _embedTokensSource = new();
        [SerializeField] private OnnxModelReference _languageModelSource = new();
        [SerializeField] private OnnxModelReference _conditionalDecoderSource = new();
        [SerializeField] private TextAsset _tokenizer;
        [SerializeField] private TextAsset _cangjieMapping;
        [SerializeField] private AudioClip _defaultVoice;

        public OnnxModelReference SpeechEncoder => _speechEncoderSource;
        public OnnxModelReference EmbedTokens => _embedTokensSource;
        public OnnxModelReference LanguageModel => _languageModelSource;
        public OnnxModelReference ConditionalDecoder => _conditionalDecoderSource;
        public TextAsset Tokenizer => _tokenizer;
        public TextAsset CangjieMapping => _cangjieMapping;
        public AudioClip DefaultVoice => _defaultVoice;
        public override string DisplayName => string.IsNullOrWhiteSpace(name) ? "Chatterbox" : name;
        public override bool IsComplete => _speechEncoderSource.IsAvailable && _embedTokensSource.IsAvailable &&
            _languageModelSource.IsAvailable && _conditionalDecoderSource.IsAvailable && _tokenizer != null && _defaultVoice != null;
        public override IOnnxModelSource[] GetAllModels() => new IOnnxModelSource[]
            { _speechEncoderSource, _embedTokensSource, _languageModelSource, _conditionalDecoderSource };

        public override ModelValidationResult Validate(ModelValidationContext context)
        {
            var result = new ModelValidationResult();
            if (!_speechEncoderSource.IsAvailable) result.Error("missing_speech_encoder", "Speech encoder model is not available.");
            if (!_embedTokensSource.IsAvailable) result.Error("missing_embeddings", "Embed tokens model is not available.");
            if (!_languageModelSource.IsAvailable) result.Error("missing_language_model", "Language model is not available.");
            if (!_conditionalDecoderSource.IsAvailable) result.Error("missing_decoder", "Conditional decoder model is not available.");
            if (_tokenizer == null) result.Error("missing_tokenizer", "Tokenizer file is not assigned.");
            if (_defaultVoice == null) result.Error("missing_voice", "Default voice audio is not assigned.");
            foreach (IOnnxModelSource model in GetAllModels()) RequireSchema(model, result);
            return ValidateCommon(context, result);
        }

#if UNITY_EDITOR
        public void SetFiles(TextAsset tokenizer, TextAsset cangjieMapping, AudioClip defaultVoice)
        {
            _tokenizer = tokenizer;
            _cangjieMapping = cangjieMapping;
            _defaultVoice = defaultVoice;
            UnityEditor.EditorUtility.SetDirty(this);
        }

        public void SetModels(OnnxModelAsset speechEncoder, OnnxModelAsset embedTokens, OnnxModelAsset languageModel,
            OnnxModelAsset conditionalDecoder, TextAsset tokenizer, TextAsset cangjieMapping, AudioClip defaultVoice,
            string modelIdentifier = null)
        {
            _speechEncoderSource.ConfigureAsset(speechEncoder); _embedTokensSource.ConfigureAsset(embedTokens);
            _languageModelSource.ConfigureAsset(languageModel); _conditionalDecoderSource.ConfigureAsset(conditionalDecoder);
            _tokenizer = tokenizer; _cangjieMapping = cangjieMapping; _defaultVoice = defaultVoice;
            UnityEditor.EditorUtility.SetDirty(this);
        }
#endif
    }
}
