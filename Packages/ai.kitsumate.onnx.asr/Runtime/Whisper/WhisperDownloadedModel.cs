using System;
using System.IO;
using System.Linq;
using KitsuMate.Onnx.Download;
using UnityEngine;

namespace KitsuMate.Onnx.Asr.Whisper
{
    /// <summary>Whisper graph contracts and runtime configuration, independent of repository and model size.</summary>
    public static class WhisperDownloadedModel
    {
        public static void Validate(DownloadedModel model)
        {
            Require(model, "mel", new[] { "audio" }, new[] { "log_mel" });
            Require(model, "encoder", Array.Empty<string>(), new[] { "last_hidden_state" });
            if (!model.GetFile("encoder").Inputs.Any(input => input.Name == "mel" || input.Name == "input_features"))
                throw new InvalidDataException("Whisper encoder requires mel or input_features input.");
            Require(model, "decoder", new[] { "input_ids", "encoder_hidden_states" }, new[] { "logits" });
            if (model.Files.Any(file => file.Role == "decoder-with-past"))
            {
                Require(model, "decoder-with-past", new[] { "input_ids" }, new[] { "logits" });
                int initial = model.GetFile("decoder").Outputs.Count(output => output.Name.StartsWith("present."));
                var cached = model.GetFile("decoder-with-past");
                if (initial == 0 || cached.Inputs.Count(input => input.Name.StartsWith("past_key_values.")) != initial ||
                    cached.Outputs.Count(output => output.Name.StartsWith("present.")) != initial)
                    throw new InvalidDataException("Whisper decoder cache contracts do not match.");
            }
            else if (!model.GetFile("decoder").Inputs.Any(input => input.Name == "use_cache_branch"))
                throw new InvalidDataException("Whisper requires a merged decoder or a decoder-with-past model.");
            model.GetFile("tokenizer");
        }

        public static WhisperEngine CreateEngine(DownloadedModel model, OnnxSessionOptions options = null)
        {
            Validate(model);
            var set = ScriptableObject.CreateInstance<WhisperModelSet>();
            set.hideFlags = HideFlags.HideAndDontSave;
            model.ConfigureModel(set.MelProcessor, "mel");
            model.ConfigureModel(set.Encoder, "encoder");
            model.ConfigureModel(set.Decoder, "decoder");
            if (model.Files.Any(file => file.Role == "decoder-with-past")) model.ConfigureModel(set.DecoderWithPast, "decoder-with-past");
            var tokenizer = new TextAsset(File.ReadAllText(model.GetPath("tokenizer"))) { hideFlags = HideFlags.HideAndDontSave };
            set.SetTokenizer(tokenizer);
            set.SetDownloadMetadata(model.Identity, Array.Empty<string>());
            var engine = ScriptableObject.CreateInstance<WhisperEngine>();
            engine.hideFlags = HideFlags.HideAndDontSave;
            engine.Configure(set, options);
            return engine;
        }

        private static void Require(DownloadedModel model, string role, string[] inputs, string[] outputs)
        {
            var file = model.GetFile(role);
            if (inputs.Any(name => !file.Inputs.Any(input => input.Name == name)) ||
                outputs.Any(name => !file.Outputs.Any(output => output.Name == name)))
                throw new InvalidDataException($"Downloaded '{role}' graph does not match the Whisper contract.");
        }
    }
}
