#if UNITY_EDITOR
using System;
using System.Linq;
using KitsuMate.Onnx.Asr.Whisper;
using KitsuMate.Onnx.Editor.Download;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Asr.Editor
{
    public static class WhisperModelDownloader
    {
        public readonly struct Source
        {
            public readonly string Name;
            public readonly string Repository;
            public readonly string Revision;
            public Source(string name, string repository, string revision)
            { Name = name; Repository = repository; Revision = revision; }
        }

        public static readonly Source[] Sources =
        {
            new Source("Whisper Tiny", "KitsuMate/whisper-tiny-onnx", "d57c18f4b7d35a780cff5edf1d6967f1297c5521"),
            new Source("Whisper Base", "KitsuMate/whisper-base-onnx", "0d4449c191a81050310938418c844d4205d9489e")
        };

        public static void Show(WhisperModelSet target, Source source)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));
            ModelDownloadWindow.Show(new ModelDownloadRequest(source.Repository, source.Revision, "whisper"),
                result => Apply(target, result));
        }

        private static void Apply(WhisperModelSet target, ModelDownloadResult result)
        {
            Validate(result);
            result.ConfigureModel(target.MelProcessor, "mel");
            result.ConfigureModel(target.Encoder, "encoder");
            result.ConfigureModel(target.Decoder, "decoder");
            if (result.ProjectPaths.ContainsKey("decoder-with-past"))
                result.ConfigureModel(target.DecoderWithPast, "decoder-with-past");
            else
                target.DecoderWithPast.Clear();
            target.SetTokenizer(result.LoadAsset<TextAsset>("tokenizer"));
            result.ApplyMetadata(target);
            AssetDatabase.SaveAssets();
            Selection.activeObject = target;
        }

        private static void Validate(ModelDownloadResult result)
        {
            result.RequireGraph("mel", new[] { "audio" }, new[] { "log_mel" });
            if (!result.GetInputNames("encoder").Any(name =>
                    name == "mel" || name == "input_features"))
                throw new InvalidOperationException(
                    "Downloaded Whisper encoder is missing input 'mel' or 'input_features'.");
            result.RequireGraph("encoder", outputs: new[] { "last_hidden_state" });
            result.RequireGraph("decoder", new[] { "input_ids", "encoder_hidden_states" },
                new[] { "logits" });

            if (!result.ProjectPaths.ContainsKey("decoder-with-past"))
            {
                if (!result.GetInputNames("decoder").Contains("use_cache_branch"))
                    throw new InvalidOperationException(
                        "A Whisper decoder without decoder-with-past must provide 'use_cache_branch'.");
                return;
            }

            result.RequireGraph("decoder-with-past", new[] { "input_ids" }, new[] { "logits" });
            int initialCacheCount = result.GetOutputNames("decoder")
                .Count(name => name.StartsWith("present.", StringComparison.Ordinal));
            int cacheInputCount = result.GetInputNames("decoder-with-past")
                .Count(name => name.StartsWith("past_key_values.", StringComparison.Ordinal));
            int cacheOutputCount = result.GetOutputNames("decoder-with-past")
                .Count(name => name.StartsWith("present.", StringComparison.Ordinal));
            if (initialCacheCount == 0 || initialCacheCount != cacheInputCount ||
                cacheInputCount != cacheOutputCount)
                throw new InvalidOperationException(
                    "Whisper initial and cached decoder cache contracts do not match.");
        }

    }
}
#endif
