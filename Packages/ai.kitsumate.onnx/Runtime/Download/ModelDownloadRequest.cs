using System;
using System.Collections.Generic;
using System.Linq;

namespace KitsuMate.Onnx.Download
{
    [Serializable]
    public sealed class ModelGraphRole
    {
        public string role;
        public string fileStem;

        public ModelGraphRole(string role, string fileStem)
        {
            this.role = role;
            this.fileStem = fileStem;
        }
    }

    /// <summary>Identifies a Hugging Face ONNX repository.</summary>
    public sealed class ModelDownloadRequest
    {
        public string Repository { get; }
        public string Revision { get; }
        public string ExpectedFamily { get; }
        public IReadOnlyList<ModelGraphRole> GraphRoles { get; }
        public IReadOnlyList<string> RequiredCompanionRoles { get; }

        public ModelDownloadRequest(string repository, string revision = "", string expectedFamily = null,
            IReadOnlyList<ModelGraphRole> graphRoles = null, IReadOnlyList<string> requiredCompanionRoles = null)
        {
            Repository = Required(repository, nameof(repository));
            Revision = revision?.Trim() ?? string.Empty;
            ExpectedFamily = expectedFamily?.Trim() ?? string.Empty;
            GraphRoles = graphRoles?.ToArray();
            RequiredCompanionRoles = requiredCompanionRoles?.ToArray();
        }

        private static string Required(string value, string name)
        {
            if (string.IsNullOrWhiteSpace(value))
                throw new ArgumentException($"{name} is required.", name);
            return value.Trim();
        }
    }
}
