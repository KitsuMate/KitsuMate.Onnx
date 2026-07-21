#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using KitsuMate.Onnx.Asr.Whisper;
using KitsuMate.Onnx.Editor.Download;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Asr.Editor
{
    /// <summary>
    /// Describes the complete Whisper artifact set accepted by the shared model downloader.
    /// The repository is deliberately user-selected: only complete, matching public ONNX exports
    /// may be installed into a Whisper model set.
    /// </summary>
    public static class WhisperModelDownloader
    {
        public static ModelDownloadRequest CreateRequest(WhisperModelSet target)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));

            return new ModelDownloadRequest
            {
                WindowTitle = "Download compatible Whisper ONNX model",
                TargetModelSet = target,
                Repositories = Array.Empty<RepositoryDefinition>(),
                Variants = new[]
                {
                    new ModelVariantDefinition(
                        "Compatible Whisper ONNX",
                        new FileDefinition("mel", "onnx/mel.onnx", "mel.onnx", "onnx/LogMelSpectro.onnx", "LogMelSpectro.onnx"),
                        new FileDefinition("encoder", "onnx/encoder_model.onnx", "encoder_model.onnx", "onnx/encoder_model_*.onnx", "encoder_model_*.onnx"),
                        new FileDefinition("decoder", "onnx/decoder_model_merged.onnx", "decoder_model_merged.onnx", "onnx/decoder_model_merged_*.onnx", "decoder_model_merged_*.onnx"),
                        new FileDefinition("tokenizer", "tokenizer.json", "onnx/tokenizer.json"))
                },
                OnFilesDownloaded = files => AssignDownloadedAssets(target, files)
            };
        }

        private static void AssignDownloadedAssets(WhisperModelSet target, Dictionary<string, string> files)
        {
            if (!files.TryGetValue("mel", out string melPath) ||
                !files.TryGetValue("encoder", out string encoderPath) ||
                !files.TryGetValue("decoder", out string decoderPath) ||
                !files.TryGetValue("tokenizer", out string tokenizerPath))
            {
                throw new InvalidOperationException("Whisper installation completed without its required model files.");
            }

            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            OnnxModelAsset mel = LoadModel(melPath, "mel");
            OnnxModelAsset encoder = LoadModel(encoderPath, "encoder");
            OnnxModelAsset decoder = LoadModel(decoderPath, "decoder");
            TextAsset tokenizer = AssetDatabase.LoadAssetAtPath<TextAsset>(tokenizerPath);
            if (tokenizer == null)
                throw new InvalidOperationException($"Downloaded Whisper tokenizer was not imported at '{tokenizerPath}'.");

            target.SetModels(mel, encoder, decoder, tokenizer);
            Selection.activeObject = target;
        }

        private static OnnxModelAsset LoadModel(string path, string key)
        {
            OnnxModelAsset model = AssetDatabase.LoadAssetAtPath<OnnxModelAsset>(path);
            if (model == null)
                throw new InvalidOperationException($"Downloaded Whisper {key} model was not imported at '{path}'.");
            return model;
        }
    }
}
#endif
