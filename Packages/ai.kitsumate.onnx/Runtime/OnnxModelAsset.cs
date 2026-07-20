using System;
using System.Collections.Generic;
using UnityEngine;

namespace KitsuMate.Onnx
{
    /// <summary>
    /// Represents an imported ONNX model plus descriptive metadata for use inside Unity.
    /// </summary>
    public class OnnxModelAsset : ScriptableObject, IOnnxModelSource
    {
        public enum MetadataInspectionState
        {
            NotInspected,
            Succeeded,
            Failed
        }

        [Serializable]
        public struct MetadataEntry
        {
            public string key;
            public string value;

            public MetadataEntry(string key, string value)
            {
                this.key = key;
                this.value = value;
            }
            
            /// <summary>Key (uppercase accessor).</summary>
            public string Key => key;
            /// <summary>Value (uppercase accessor).</summary>
            public string Value => value;
        }
        
        /// <summary>
        /// Information about a tensor input or output.
        /// </summary>
        [Serializable]
        public struct TensorInfo
        {
            public string name;
            public string elementType;
            public int[] shape;
            public string shapeDescription;

            public TensorInfo(string name, string elementType, int[] shape, string shapeDescription = null)
            {
                this.name = name;
                this.elementType = elementType;
                this.shape = shape ?? Array.Empty<int>();
                this.shapeDescription = shapeDescription ?? (shape != null && shape.Length > 0 
                    ? string.Join("x", shape) : "?");
            }
            
            /// <summary>Name (uppercase accessor).</summary>
            public string Name => name;
            /// <summary>Element type (uppercase accessor).</summary>
            public string ElementType => elementType;
            /// <summary>Shape array (uppercase accessor).</summary>
            public int[] Shape => shape;
            /// <summary>Shape description (uppercase accessor).</summary>
            public string ShapeDescription => shapeDescription;
            
            public override string ToString()
            {
                var shapeStr = shape != null && shape.Length > 0 
                    ? $"[{string.Join(", ", shape)}]" 
                    : "[]";
                return $"{name}: {elementType}{shapeStr}";
            }
        }

        [SerializeField] private string _modelName = string.Empty;
        [SerializeField] private string _graphName = string.Empty;
        [SerializeField] private string _domain = string.Empty;
        [SerializeField] private string _description = string.Empty;
        [SerializeField] private string _producer = string.Empty;
        [SerializeField] private long _opsetVersion;
        [SerializeField] private long _fileSizeBytes;
        [SerializeField] private string _importedUtc = string.Empty;
        [SerializeField] private string _sourceAssetPath = string.Empty;
        [SerializeField] private List<TensorInfo> _inputs = new();
        [SerializeField] private List<TensorInfo> _outputs = new();
        [SerializeField] private List<MetadataEntry> _customMetadata = new();
        [SerializeField] private OnnxOptimizationLevel _optimizationLevel = OnnxOptimizationLevel.Extended;
        [SerializeField, HideInInspector] private OnnxModelData _modelDataAsset;
        [SerializeField] private MetadataInspectionState _metadataState = MetadataInspectionState.NotInspected;
        [SerializeField] private string _metadataError = string.Empty;
        [SerializeField] private bool _externalDataMerged;

        public string ModelName => _modelName;
        public string GraphName => _graphName;
        public string Domain => _domain;
        public string Description => _description;
        public string Producer => _producer;
        public long OpsetVersion => _opsetVersion;
        public long FileSizeBytes => _fileSizeBytes;
        public string ImportedUtc => _importedUtc;
        public string SourceAssetPath => _sourceAssetPath;
        public IReadOnlyList<TensorInfo> Inputs => _inputs;
        public IReadOnlyList<TensorInfo> Outputs => _outputs;
        public IReadOnlyList<MetadataEntry> CustomMetadata => _customMetadata;
        
        /// <summary>Formatted file size string.</summary>
        public string FileSizeFormatted => FormatFileSize(_fileSizeBytes);
        
        /// <summary>Default optimization level for sessions.</summary>
        public OnnxOptimizationLevel DefaultOptimizationLevel
        {
            get => _optimizationLevel;
            set => _optimizationLevel = value;
        }
        
        public MetadataInspectionState MetadataState => _metadataState;
        public string MetadataError => _metadataError;
        
        /// <summary>Whether external data was merged into this asset at import time.</summary>
        public bool ExternalDataMerged => _externalDataMerged;
        
        /// <summary>Reference to the model data sub-asset.</summary>
        public OnnxModelData ModelDataAsset => _modelDataAsset;
        
        /// <summary>Whether this asset has model data available.</summary>
        public bool HasModelData => _modelDataAsset != null && _modelDataAsset.HasData;
        
        /// <summary>Size of embedded model data in bytes.</summary>
        public long EmbeddedSizeBytes => _modelDataAsset?.Size ?? 0;
        
        /// <summary>Formatted embedded size string.</summary>
        public string EmbeddedSizeFormatted => FormatFileSize(EmbeddedSizeBytes);

        /// <summary>Gets the model bytes as ReadOnlyMemory.</summary>
        public ReadOnlyMemory<byte> ModelData => _modelDataAsset?.Data ?? ReadOnlyMemory<byte>.Empty;
        public string SourceName => name;
        public bool IsAvailable => HasResolvableData;
        public bool HasInspectedMetadata => _metadataState == MetadataInspectionState.Succeeded;
        public bool IsMetadataStale => false;
        public OnnxModelAsset ImportedAsset => this;
        public bool HasResolvableData => HasModelData;

        public string ResolveModelPath()
        {
            return ResolveSourcePath();
        }

        /// <summary>Gets a copy of the model bytes.</summary>
        public byte[] CopyModelData()
        {
            return _modelDataAsset?.GetDataCopy() ?? Array.Empty<byte>();
        }

        public void Populate(
            string modelName,
            string graphName,
            string domain,
            string description,
            string producer,
            long opsetVersion,
            long fileSizeBytes,
            DateTime importedUtc,
            string sourceAssetPath,
            IEnumerable<TensorInfo> inputs,
            IEnumerable<TensorInfo> outputs,
            IEnumerable<MetadataEntry> metadata,
            OnnxModelData modelDataAsset,
            OnnxOptimizationLevel? optimizationLevel = null,
            MetadataInspectionState metadataState = MetadataInspectionState.NotInspected,
            string metadataError = null,
            bool externalDataMerged = false)
        {
            _modelName = modelName ?? string.Empty;
            _graphName = graphName ?? string.Empty;
            _domain = domain ?? string.Empty;
            _description = description ?? string.Empty;
            _producer = producer ?? string.Empty;
            _opsetVersion = opsetVersion;
            _fileSizeBytes = fileSizeBytes;
            _importedUtc = importedUtc.ToUniversalTime().ToString("u");
            _sourceAssetPath = sourceAssetPath ?? string.Empty;

            _inputs.Clear();
            if (inputs != null)
                _inputs.AddRange(inputs);

            _outputs.Clear();
            if (outputs != null)
                _outputs.AddRange(outputs);

            _customMetadata.Clear();
            if (metadata != null)
                _customMetadata.AddRange(metadata);

            _modelDataAsset = modelDataAsset;
            _optimizationLevel = optimizationLevel ?? OnnxOptimizationLevel.Extended;
            _metadataState = metadataState;
            _metadataError = metadataError ?? string.Empty;
            _externalDataMerged = externalDataMerged;
        }

        public void ApplyMetadata(
            string modelName,
            string graphName,
            string domain,
            string description,
            string producer,
            long opsetVersion,
            IEnumerable<TensorInfo> inputs,
            IEnumerable<TensorInfo> outputs,
            IEnumerable<MetadataEntry> metadata,
            MetadataInspectionState metadataState,
            string metadataError)
        {
            if (!string.IsNullOrEmpty(modelName))
                _modelName = modelName;

            _graphName = graphName ?? string.Empty;
            _domain = domain ?? string.Empty;
            _description = description ?? string.Empty;
            _producer = producer ?? string.Empty;
            _opsetVersion = opsetVersion;

            _inputs.Clear();
            if (inputs != null)
                _inputs.AddRange(inputs);

            _outputs.Clear();
            if (outputs != null)
                _outputs.AddRange(outputs);

            _customMetadata.Clear();
            if (metadata != null)
                _customMetadata.AddRange(metadata);

            _metadataState = metadataState;
            _metadataError = metadataError ?? string.Empty;
        }

        /// <summary>
        /// Resolves the absolute path to the source ONNX file.
        /// </summary>
        public string ResolveSourcePath()
        {
            if (string.IsNullOrEmpty(_sourceAssetPath))
                return null;
                
            if (System.IO.Path.IsPathRooted(_sourceAssetPath))
                return _sourceAssetPath;
                
            // Resolve relative to project root
            string projectRoot = System.IO.Directory.GetCurrentDirectory();
            return System.IO.Path.GetFullPath(System.IO.Path.Combine(projectRoot, _sourceAssetPath));
        }

        private static string FormatFileSize(long bytes)
        {
            string[] sizes = { "B", "KB", "MB", "GB", "TB" };
            int order = 0;
            double size = bytes;
            while (size >= 1024 && order < sizes.Length - 1)
            {
                order++;
                size /= 1024;
            }
            return $"{size:0.##} {sizes[order]}";
        }
    }
}
