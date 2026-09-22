using KitsuMate.Onnx.Download;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    [CreateAssetMenu(fileName = "ChatterboxModelSet", menuName = "KitsuMate/ONNX/TTS/Chatterbox Model Set")]
    public class ChatterboxModelSet : StandardModelSet
    {
        public override System.Collections.Generic.IReadOnlyList<ModelGraphRole> DownloadGraphRoles => new[]
        {
            new ModelGraphRole("speech-encoder", "speech_encoder"),
            new ModelGraphRole("embed-tokens", "embed_tokens"),
            new ModelGraphRole("language-model", "language_model"),
            new ModelGraphRole("conditional-decoder", "conditional_decoder")
        };
        public override string[] DownloadCompanionRoles => new[]
            { "tokenizer", "cangjie", "japanese-readings", "russian-stress", "chinese-words", "voice" };
        public override string[] DownloadRequiredCompanionRoles => new[] { "tokenizer" };
        public override bool ValidateDownloadedBinding => true;

        public override TextFileReference[] GetAllTextFiles() => new[]
            { _tokenizer, _cangjieMapping, _japaneseReadings, _russianStress, _chineseWords };

        public override System.Collections.Generic.IEnumerable<(string Family, string Repository)> RepositorySuggestions
        {
            get
            {
                yield return ("chatterbox", "KitsuMate/chatterbox-multilingual-ONNX");
                yield return ("chatterbox", "KitsuMate/chatterbox-turbo-onnx");
                yield return ("chatterbox", "KitsuMate/chatterbox-nano-onnx");
            }
        }

        protected override async Task BindInstallationAsync(DownloadedModel installation, ResolvedModelSet resources, CancellationToken cancellationToken)
        {
            installation.ConfigureModel(_speechEncoderSource, "speech-encoder");
            installation.ConfigureModel(_embedTokensSource, "embed-tokens");
            installation.ConfigureModel(_languageModelSource, "language-model");
            installation.ConfigureModel(_conditionalDecoderSource, "conditional-decoder");
            _tokenizer = resources.ReadText(installation, "tokenizer");
            _cangjieMapping = resources.ReadText(installation, "cangjie", optional: true);
            _japaneseReadings = resources.ReadText(installation, "japanese-readings", optional: true);
            _russianStress = resources.ReadText(installation, "russian-stress", optional: true);
            _chineseWords = resources.ReadText(installation, "chinese-words", optional: true);
            if (installation.Files.Any(file => file.Role == "voice"))
                _defaultVoice = await resources.ReadAudioAsync(installation, "voice", cancellationToken);
        }

        [SerializeField] private OnnxModelReference _speechEncoderSource = new();
        [SerializeField] private OnnxModelReference _embedTokensSource = new();
        [SerializeField] private OnnxModelReference _languageModelSource = new();
        [SerializeField] private OnnxModelReference _conditionalDecoderSource = new();
        [SerializeField] private TextFileReference _tokenizer = new();
        [SerializeField] private TextFileReference _cangjieMapping = new();
        [SerializeField] private TextFileReference _japaneseReadings = new();
        [SerializeField] private TextFileReference _russianStress = new();
        [SerializeField] private TextFileReference _chineseWords = new();
        [SerializeField] private AudioClip _defaultVoice;
        [SerializeField] private ChatterboxGenerationConfig _generation = new();
        public ChatterboxGenerationConfig Generation => _generation;

        public OnnxModelReference SpeechEncoder => _speechEncoderSource;
        public OnnxModelReference EmbedTokens => _embedTokensSource;
        public OnnxModelReference LanguageModel => _languageModelSource;
        public OnnxModelReference ConditionalDecoder => _conditionalDecoderSource;
        public TextFileReference Tokenizer => _tokenizer?.IsAvailable == true ? _tokenizer : null;
        public TextFileReference CangjieMapping => _cangjieMapping?.IsAvailable == true ? _cangjieMapping : null;
        public TextFileReference JapaneseReadings => _japaneseReadings?.IsAvailable == true ? _japaneseReadings : null;
        public TextFileReference RussianStress => _russianStress?.IsAvailable == true ? _russianStress : null;
        public TextFileReference ChineseWords => _chineseWords?.IsAvailable == true ? _chineseWords : null;
        public AudioClip DefaultVoice => _defaultVoice;
        public override string DisplayName => string.IsNullOrWhiteSpace(name) ? "Chatterbox" : name;
        public override bool IsComplete => _speechEncoderSource.IsAvailable && _embedTokensSource.IsAvailable &&
            _languageModelSource.IsAvailable && _conditionalDecoderSource.IsAvailable && _tokenizer?.IsAvailable == true;
        public override IOnnxModelSource[] GetAllModels() => new IOnnxModelSource[]
            { _speechEncoderSource, _embedTokensSource, _languageModelSource, _conditionalDecoderSource };

        public override ModelValidationResult Validate(ModelValidationContext context)
        {
            var result = new ModelValidationResult();
            if (!_speechEncoderSource.IsAvailable) result.Error("missing_speech_encoder", "Speech encoder model is not available.");
            if (!_embedTokensSource.IsAvailable) result.Error("missing_embeddings", "Embed tokens model is not available.");
            if (!_languageModelSource.IsAvailable) result.Error("missing_language_model", "Language model is not available.");
            if (!_conditionalDecoderSource.IsAvailable) result.Error("missing_decoder", "Conditional decoder model is not available.");
            if (_tokenizer?.IsAvailable != true) result.Error("missing_tokenizer", "Tokenizer file is not assigned.");
            foreach (IOnnxModelSource model in GetAllModels()) RequireSchema(model, result);
            return ValidateCommon(context, result);
        }

#if UNITY_EDITOR
        public void SetFiles(TextFileReference tokenizer, TextFileReference cangjieMapping,
            TextFileReference japaneseReadings, TextFileReference russianStress, TextFileReference chineseWords,
            AudioClip defaultVoice)
        {
            _tokenizer = tokenizer;
            _cangjieMapping = cangjieMapping;
            _japaneseReadings = japaneseReadings;
            _russianStress = russianStress;
            _chineseWords = chineseWords;
            _defaultVoice = defaultVoice;
            UnityEditor.EditorUtility.SetDirty(this);
        }

        public void SetModels(OnnxModelAsset speechEncoder, OnnxModelAsset embedTokens, OnnxModelAsset languageModel,
            OnnxModelAsset conditionalDecoder, TextFileReference tokenizer, TextFileReference cangjieMapping,
            TextFileReference japaneseReadings, TextFileReference russianStress, TextFileReference chineseWords,
            AudioClip defaultVoice,
            string modelIdentifier = null)
        {
            _speechEncoderSource.ConfigureAsset(speechEncoder); _embedTokensSource.ConfigureAsset(embedTokens);
            _languageModelSource.ConfigureAsset(languageModel); _conditionalDecoderSource.ConfigureAsset(conditionalDecoder);
            _tokenizer = tokenizer; _cangjieMapping = cangjieMapping; _japaneseReadings = japaneseReadings;
            _russianStress = russianStress; _chineseWords = chineseWords; _defaultVoice = defaultVoice;
            UnityEditor.EditorUtility.SetDirty(this);
        }

#endif
    }
}
