using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using KitsuMate.Onnx.Download;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    [CreateAssetMenu(fileName = "ChatterboxSplitModelSet", menuName = "KitsuMate/ONNX/TTS/Chatterbox Split Model Set")]
    public sealed class ChatterboxSplitModelSet : StandardModelSet
    {
        [SerializeField] private OnnxModelReference speechEncoder = new();
        [SerializeField] private OnnxModelReference embeddingLanguageModel = new();
        [SerializeField] private OnnxModelReference flowPrepare = new();
        [SerializeField] private OnnxModelReference flowStep = new();
        [SerializeField] private OnnxModelReference vocoder = new();
        [SerializeField] private TextFileReference tokenizer = new();
        [SerializeField] private TextFileReference cangjieMapping = new();
        [SerializeField] private TextFileReference japaneseReadings = new();
        [SerializeField] private TextFileReference russianStress = new();
        [SerializeField] private TextFileReference chineseWords = new();
        [SerializeField] private AudioClip defaultVoice;
        [SerializeField] private ChatterboxGenerationConfig generation = new();
        [SerializeField, Range(1, 20)] private int flowSteps = 6;
        [SerializeField, Range(0, 2)] private float flowGuidance = 0.7f;

        private static readonly ModelGraphRole[] Roles =
        {
            new("speech-encoder", "speech_encoder_slim"),
            new("embedding-language-model", "embedding_language_model"),
            new("flow-prepare", "flow_prepare"),
            new("flow-step", "flow_step"),
            new("vocoder", "vocoder")
        };

        public override IReadOnlyList<ModelGraphRole> DownloadGraphRoles => Roles;
        public override string[] DownloadCompanionRoles => new[]
            { "tokenizer", "cangjie", "japanese-readings", "russian-stress", "chinese-words", "voice" };
        public override string[] DownloadRequiredCompanionRoles => new[] { "tokenizer" };
        public override bool ValidateDownloadedBinding => true;
        public override IEnumerable<(string Family, string Repository)> RepositorySuggestions
        {
            get { yield return ("chatterbox", "KitsuMate/chatterbox-multilingual-v3-onnx"); }
        }

        public OnnxModelReference SpeechEncoder => speechEncoder;
        public OnnxModelReference EmbeddingLanguageModel => embeddingLanguageModel;
        public OnnxModelReference FlowPrepare => flowPrepare;
        public OnnxModelReference FlowStep => flowStep;
        public OnnxModelReference Vocoder => vocoder;
        public TextFileReference Tokenizer => tokenizer?.IsAvailable == true ? tokenizer : null;
        public TextFileReference CangjieMapping => cangjieMapping?.IsAvailable == true ? cangjieMapping : null;
        public TextFileReference JapaneseReadings => japaneseReadings?.IsAvailable == true ? japaneseReadings : null;
        public TextFileReference RussianStress => russianStress?.IsAvailable == true ? russianStress : null;
        public TextFileReference ChineseWords => chineseWords?.IsAvailable == true ? chineseWords : null;
        public AudioClip DefaultVoice => defaultVoice;
        public ChatterboxGenerationConfig Generation => generation;
        public int FlowSteps => flowSteps;
        public float FlowGuidance => flowGuidance;
        public override string DisplayName => string.IsNullOrWhiteSpace(name) ? "Chatterbox Split" : name;
        public override bool IsComplete => speechEncoder.IsAvailable && embeddingLanguageModel.IsAvailable &&
            flowPrepare.IsAvailable && flowStep.IsAvailable && vocoder.IsAvailable && Tokenizer != null;
        public override IOnnxModelSource[] GetAllModels() =>
            new IOnnxModelSource[] { speechEncoder, embeddingLanguageModel, flowPrepare, flowStep, vocoder };
        public override TextFileReference[] GetAllTextFiles() => new[]
            { tokenizer, cangjieMapping, japaneseReadings, russianStress, chineseWords };

        protected override async Task BindInstallationAsync(DownloadedModel installation, ResolvedModelSet resources,
            CancellationToken cancellationToken)
        {
            installation.ConfigureModel(speechEncoder, "speech-encoder");
            installation.ConfigureModel(embeddingLanguageModel, "embedding-language-model");
            installation.ConfigureModel(flowPrepare, "flow-prepare");
            installation.ConfigureModel(flowStep, "flow-step");
            installation.ConfigureModel(vocoder, "vocoder");
            tokenizer = resources.ReadText(installation, "tokenizer");
            cangjieMapping = resources.ReadText(installation, "cangjie", optional: true);
            japaneseReadings = resources.ReadText(installation, "japanese-readings", optional: true);
            russianStress = resources.ReadText(installation, "russian-stress", optional: true);
            chineseWords = resources.ReadText(installation, "chinese-words", optional: true);
            if (installation.Files.Any(file => file.Role == "voice"))
                defaultVoice = await resources.ReadAudioAsync(installation, "voice", cancellationToken);
        }

        public override ModelValidationResult Validate(ModelValidationContext context)
        {
            var result = new ModelValidationResult();
            foreach (var (source, role) in new[]
            {
                (speechEncoder, "speech encoder"), (embeddingLanguageModel, "embedding language model"),
                (flowPrepare, "flow prepare"), (flowStep, "flow step"), (vocoder, "vocoder")
            })
            {
                if (!source.IsAvailable) result.Error("missing_graph", $"Chatterbox split {role} is not available.");
                RequireSchema(source, result);
            }
            if (Tokenizer == null) result.Error("missing_tokenizer", "Tokenizer file is not assigned.");
            RequireInput(speechEncoder, "audio_values", result);
            RequireOutput(speechEncoder, "audio_features", result);
            RequireOutput(speechEncoder, "audio_tokens", result);
            RequireOutput(speechEncoder, "speaker_embeddings", result);
            RequireOutput(speechEncoder, "speaker_features", result);
            foreach (string input in new[] { "input_ids", "token_position_ids", "conditioning",
                "text_conditioning", "attention_mask", "lm_position_ids" })
                RequireInput(embeddingLanguageModel, input, result);
            RequireOutput(embeddingLanguageModel, "logits", result);
            foreach (string input in new[] { "speech_tokens", "speaker_embeddings", "speaker_features" })
                RequireInput(flowPrepare, input, result);
            foreach (string output in new[] { "x", "mask", "mu", "speakers", "cond", "prefix" })
                RequireOutput(flowPrepare, output, result);
            foreach (string input in new[] { "x", "mask", "mu", "speakers", "cond", "time", "delta", "guidance" })
                RequireInput(flowStep, input, result);
            RequireOutput(flowStep, "next_x", result);
            RequireInput(vocoder, "x", result);
            RequireInput(vocoder, "prefix", result);
            RequireOutput(vocoder, "waveform", result);
            if (flowSteps < 1 || flowSteps > 20 || !float.IsFinite(flowGuidance) || flowGuidance < 0)
                result.Error("invalid_flow", "Split flow requires 1-20 steps and nonnegative finite guidance.");
            return ValidateCommon(context, result);
        }
    }
}
