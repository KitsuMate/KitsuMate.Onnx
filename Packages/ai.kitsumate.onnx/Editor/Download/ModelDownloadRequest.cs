using System;

namespace KitsuMate.Onnx.Editor.Download
{
    /// <summary>Identifies a Hugging Face ONNX repository.</summary>
    public sealed class ModelDownloadRequest
    {
        public string Repository { get; }
        public string Revision { get; }
        public string ExpectedFamily { get; }

        public ModelDownloadRequest(string repository, string revision, string expectedFamily = null)
        {
            Repository = Required(repository, nameof(repository));
            Revision = Required(revision, nameof(revision));
            ExpectedFamily = expectedFamily?.Trim() ?? string.Empty;
        }

        private static string Required(string value, string name)
        {
            if (string.IsNullOrWhiteSpace(value))
                throw new ArgumentException($"{name} is required.", name);
            return value.Trim();
        }
    }
}
