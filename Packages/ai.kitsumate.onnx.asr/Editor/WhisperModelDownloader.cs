#if UNITY_EDITOR
using System;
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
            new Source("Whisper Tiny", "KitsuMate/whisper-tiny-onnx", "4c03eadcc2691a9c8b23e7e9fdbf1403e07fc0b9"),
            new Source("Whisper Base", "KitsuMate/whisper-base-onnx", "5405863a79f8c04cb8cfd22a5ff773af1feb39b7")
        };

        public static void Show(WhisperModelSet target, Source source)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));
            ModelDownloadWindow.Show(new ModelDownloadRequest(source.Repository, source.Revision, string.Empty,
                ModelRoot(), "whisper"), result => Apply(target, result));
        }

        private static void Apply(WhisperModelSet target, ModelDownloadResult result)
        {
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

        private static string ModelRoot()
        {
            OnnxSettings settings = OnnxSettings.Load();
            return settings != null ? settings.ModelStorageRoot : "Assets/StreamingAssets/KitsuMateModels";
        }
    }
}
#endif
