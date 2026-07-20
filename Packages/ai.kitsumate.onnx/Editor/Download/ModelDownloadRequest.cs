using System;
using System.Collections.Generic;

namespace KitsuMate.Onnx.Editor.Download
{
    /// <summary>
    /// Defines a file to download with multiple pattern options.
    /// The first matching pattern wins.
    /// </summary>
    [Serializable]
    public struct FileDefinition
    {
        /// <summary>Unique key for this file (e.g., "encoder", "decoder", "vocab").</summary>
        public string Key;
        
        /// <summary>
        /// Ordered list of file patterns to match.
        /// Supports wildcards (*) and path prefixes (e.g., "onnx/encoder_model.onnx").
        /// First matching pattern wins.
        /// </summary>
        public string[] Patterns;
        
        /// <summary>
        /// If true, the file is not required for variant matching.
        /// Optional files are still downloaded when found.
        /// </summary>
        public bool Optional;
        
        public FileDefinition(string key, params string[] patterns)
        {
            Key = key;
            Patterns = patterns;
            Optional = false;
        }
        
        public FileDefinition(string key, bool optional, params string[] patterns)
        {
            Key = key;
            Patterns = patterns;
            Optional = optional;
        }
    }
    
    /// <summary>
    /// Defines a model variant (e.g., different quantization levels).
    /// </summary>
    [Serializable]
    public struct ModelVariantDefinition
    {
        /// <summary>Display name for the variant (e.g., "Float32", "Int8").</summary>
        public string Name;
        
        /// <summary>Files required for this variant.</summary>
        public FileDefinition[] Files;
        
        public ModelVariantDefinition(string name, params FileDefinition[] files)
        {
            Name = name;
            Files = files;
        }
    }
    
    /// <summary>
    /// Defines a repository source for model downloads.
    /// </summary>
    [Serializable]
    public struct RepositoryDefinition
    {
        /// <summary>Display name for the repository (e.g., "Whisper Tiny (~75MB)").</summary>
        public string DisplayName;
        
        /// <summary>
        /// Repository input - can be "owner/repo" format (defaults to HuggingFace)
        /// or a full URL to any supported git platform.
        /// </summary>
        public string RepoInput;
        
        public RepositoryDefinition(string displayName, string repoInput)
        {
            DisplayName = displayName;
            RepoInput = repoInput;
        }
    }
    
    /// <summary>
    /// Request configuration for the model download window.
    /// </summary>
    public class ModelDownloadRequest
    {
        /// <summary>Title for the download window.</summary>
        public string WindowTitle { get; set; } = "Download Models";
        
        /// <summary>Available repositories to download from.</summary>
        public RepositoryDefinition[] Repositories { get; set; }
        
        /// <summary>Available variants with their file patterns.</summary>
        public ModelVariantDefinition[] Variants { get; set; }
        
        /// <summary>
        /// Optional factory that builds variants dynamically from the fetched file list.
        /// Called after file listing completes. When set, overrides <see cref="Variants"/>.
        /// </summary>
        public Func<List<RepoFile>, ModelVariantDefinition[]> VariantBuilder { get; set; }
        
        /// <summary>Target model set to assign downloaded files to.</summary>
        public ModelSet TargetModelSet { get; set; }
        
        /// <summary>
        /// Callback invoked when download completes successfully.
        /// Dictionary maps file keys to their downloaded asset paths.
        /// </summary>
        public Action<Dictionary<string, string>> OnFilesDownloaded { get; set; }
        
        /// <summary>
        /// If true, existing files will be replaced.
        /// If false, only missing files will be downloaded.
        /// </summary>
        public bool ReplaceExisting { get; set; } = true;
    }
}
