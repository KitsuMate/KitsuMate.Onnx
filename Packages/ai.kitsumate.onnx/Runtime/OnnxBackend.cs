using System;
using System.Collections.Generic;
using UnityEngine;

namespace KitsuMate.Onnx
{
    /// <summary>
    /// Options for configuring ONNX inference sessions.
    /// </summary>
    /// <summary>Portable optimization intent understood by individual backend implementations.</summary>
    public enum OnnxOptimizationLevel
    {
        DisableAll,
        Basic,
        Extended,
        All
    }

    [Serializable]
    public class OnnxSessionOptions
    {
        public OnnxOptimizationLevel OptimizationLevel = OnnxOptimizationLevel.All;
        public bool EnableMemoryPattern = true;
        public bool EnableCpuMemArena = true;
        public int IntraOpThreads = 0;
        public int InterOpThreads = 0;
        
        public static OnnxSessionOptions Default => new();
    }
    
    /// <summary>
    /// Abstraction layer for different ONNX backends.
    /// Allows switching between ONNX Runtime, Unity Sentis, WebGL, etc.
    /// </summary>
    public abstract class OnnxBackend : ScriptableObject, IDisposable
    {
        /// <summary>Stable identifier used by model compatibility declarations.</summary>
        public virtual string BackendId => GetType().FullName;

        /// <summary>Display name for this backend.</summary>
        public abstract string DisplayName { get; }
        
        /// <summary>Platforms this backend supports.</summary>
        public abstract IReadOnlyList<RuntimePlatform> SupportedPlatforms { get; }
        
        /// <summary>Whether this backend is available on the current platform.</summary>
        public abstract bool IsAvailable { get; }
        
        /// <summary>Priority for automatic backend selection (higher = preferred).</summary>
        public abstract int Priority { get; }
        
        /// <summary>
        /// Creates an inference session from a model asset.
        /// Uses a direct reference to model bytes (zero-copy).
        /// </summary>
        public virtual IOnnxSession CreateSession(OnnxModelAsset model)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            if (!model.HasModelData) throw new InvalidOperationException($"ONNX model '{model.name}' does not contain any byte data.");
            return CreateSession(model.ModelDataAsset.GetDataRef(), OnnxSessionOptions.Default);
        }

        public virtual IOnnxSession CreateSession(IOnnxModelSource source)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (source.ImportedAsset != null) return CreateSession(source.ImportedAsset);
            string path = source.ResolveModelPath();
            if (string.IsNullOrWhiteSpace(path) || !System.IO.File.Exists(path))
                throw new InvalidOperationException($"ONNX model source '{source.SourceName}' could not be resolved.");
            return CreateSession(path, OnnxSessionOptions.Default);
        }
        
        /// <summary>
        /// Creates an inference session from model bytes.
        /// </summary>
        public IOnnxSession CreateSession(byte[] modelData)
        {
            return CreateSession(modelData, OnnxSessionOptions.Default);
        }

        /// <summary>
        /// Creates a session directly from an ONNX file. Prefer this for large models so
        /// Unity does not retain a duplicate managed byte array. Relative paths are rejected.
        /// </summary>
        public IOnnxSession CreateSession(string modelPath)
        {
            return CreateSession(modelPath, OnnxSessionOptions.Default);
        }
        
        /// <summary>
        /// Creates an inference session with custom options.
        /// </summary>
        public abstract IOnnxSession CreateSession(byte[] modelData, OnnxSessionOptions options);

        /// <summary>Creates an inference session from an absolute model path.</summary>
        public abstract IOnnxSession CreateSession(string modelPath, OnnxSessionOptions options);
        
        /// <summary>Disposes of backend resources.</summary>
        public virtual void Dispose() { }
        
    }
}
