using System;
using System.IO;
using System.Linq;
using KitsuMate.Onnx.Download;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings
{
    [Serializable]
    public sealed class TextEmbeddingConfiguration
    {
        public int Dimension = 384;
        public int MaxSequenceLength = 512;
        public bool Normalize = true;
        public string QueryPrefix = "";
        public string DocumentPrefix = "";
    }

    public static class TextEmbeddingDownloadedModel
    {
        public static void Validate(DownloadedModel model)
        {
            var graph = model.GetFile("model");
            if (!graph.Inputs.Any(input => input.Name == "input_ids") ||
                !graph.Inputs.Any(input => input.Name == "attention_mask") ||
                !graph.Outputs.Any(output => output.Name == "sentence_embedding" || output.Name == "last_hidden_state"))
                throw new InvalidDataException("Downloaded graph does not match the text embedding contract.");
            model.GetFile("tokenizer");
        }

        public static TextEmbeddingEngine CreateEngine(DownloadedModel model, TextEmbeddingConfiguration configuration,
            OnnxSessionOptions options = null)
        {
            Validate(model);
            var set = ScriptableObject.CreateInstance<TextEmbeddingModelSet>();
            set.hideFlags = HideFlags.HideAndDontSave;
            model.ConfigureModel(set.EmbeddingModel, "model");
            set.ConfigureDownloaded(model.DirectoryPath, configuration);
            set.SetDownloadMetadata(model.Identity, Array.Empty<string>());
            var engine = ScriptableObject.CreateInstance<TextEmbeddingEngine>();
            engine.hideFlags = HideFlags.HideAndDontSave;
            engine.Configure(set, options);
            return engine;
        }
    }
}
