using System;

namespace KitsuMate.Onnx.Editor.Download
{
    /// <summary>
    /// Identifies one model variant in a Hugging Face ONNX repository.
    /// </summary>
    public sealed class ModelDownloadRequest
    {
        public string Repository { get; }
        public string Revision { get; }
        public string Variant { get; }
        public string Destination { get; }
        public string ExpectedFamily { get; }

        public ModelDownloadRequest(string repository, string revision, string variant, string destination,
            string expectedFamily = null)
        {
            Repository = Required(repository, nameof(repository));
            Revision = Required(revision, nameof(revision));
            Variant = variant?.Trim() ?? string.Empty;
            Destination = Required(destination, nameof(destination));
            ExpectedFamily = expectedFamily?.Trim() ?? string.Empty;
        }

        public ModelDownloadRequest WithVariant(string variant)
        {
            return new ModelDownloadRequest(Repository, Revision, variant, Destination, ExpectedFamily);
        }

        private static string Required(string value, string name)
        {
            if (string.IsNullOrWhiteSpace(value))
                throw new ArgumentException($"{name} is required.", name);
            return value.Trim();
        }
    }
}
