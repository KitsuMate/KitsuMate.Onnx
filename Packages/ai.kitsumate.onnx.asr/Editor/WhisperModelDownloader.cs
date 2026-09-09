#if UNITY_EDITOR
using System;
using System.Linq;
using KitsuMate.Onnx.Asr.Whisper;
using KitsuMate.Onnx.Editor.Download;
using KitsuMate.Onnx.Download;
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

        private static void Validate(ModelDownloadResult result) => WhisperDownloadedModel.Validate(result.Downloaded);
    }
}
#endif
