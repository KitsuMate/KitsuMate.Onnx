using System;
using System.Collections.Generic;
using System.Linq;

namespace KitsuMate.Onnx.Download
{
    /// <summary>Application-supplied defaults; no repository is selected by the library.</summary>
    [Serializable]
    public sealed class ModelDownloadProfile
    {
        public string displayName;
        public string repository;
        public string revision = "main";
        public string family;
        public ArtifactSelection[] artifacts = Array.Empty<ArtifactSelection>();

        [Serializable]
        public sealed class ArtifactSelection { public string role; public string path; }

        public ModelDownloadRequest CreateRequest() => new(repository, revision, family);
        public IReadOnlyDictionary<string, string> CreateSelection() => artifacts.ToDictionary(item => item.role, item => item.path);
    }
}
