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
            new Source("Whisper Tiny", "KitsuMate/whisper-tiny-onnx", "2e633013560a290f0520dadb295d782e8cb092b1"),
            new Source("Whisper Base", "KitsuMate/whisper-base-onnx", "3a1dbc0e5f9b1b00d4c3a2aa82919867654113f7")
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
