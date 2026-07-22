using System;
using System.Collections.Generic;
using UnityEngine;

namespace KitsuMate.Onnx
{
    /// <summary>
    /// Project-wide ONNX settings for backend configuration.
    /// </summary>
    public class OnnxSettings : ScriptableObject
    {
        [Serializable]
        public struct PlatformOverride
        {
            public RuntimePlatform Platform;
            public OnnxBackend Backend;
        }
        
        [SerializeField] 
        private OnnxBackend _defaultBackend;
        
        [SerializeField] 
        private List<PlatformOverride> _platformOverrides = new();
        
        [SerializeField] 
        private bool _verboseLogging;
        [SerializeField] private string _modelStorageRoot = "Assets/StreamingAssets/KitsuMateModels";
        [SerializeField] private List<InferenceEngineBase> _defaultEngines = new();
        
        /// <summary>Default ONNX backend.</summary>
        public OnnxBackend DefaultBackend
        {
            get => _defaultBackend;
            set => _defaultBackend = value;
        }
        
        /// <summary>Platform-specific backend overrides.</summary>
        public IReadOnlyList<PlatformOverride> PlatformOverrides => _platformOverrides;
        
        /// <summary>Enable verbose logging for debugging.</summary>
        public bool VerboseLogging
        {
            get => _verboseLogging;
            set => _verboseLogging = value;
        }
        public string ModelStorageRoot => string.IsNullOrWhiteSpace(_modelStorageRoot) ? "Assets/StreamingAssets/KitsuMateModels" : _modelStorageRoot;
        public TEngine GetDefaultEngine<TEngine>() where TEngine : InferenceEngineBase
        {
            foreach (InferenceEngineBase engine in _defaultEngines) if (engine is TEngine typed) return typed;
            return null;
        }

#if UNITY_EDITOR
        public bool RegisterDefaultEngine(InferenceEngineBase engine)
        {
            if (engine == null) throw new ArgumentNullException(nameof(engine));
            if (_defaultEngines.Exists(existing => existing != null && existing.GetType() == engine.GetType()))
                return false;
            _defaultEngines.Add(engine);
            UnityEditor.EditorUtility.SetDirty(this);
            return true;
        }
#endif
        
        /// <summary>
        /// Gets the backend for the current platform.
        /// </summary>
        public OnnxBackend GetBackendForCurrentPlatform()
        {
            return GetBackendForPlatform(Application.platform);
        }
        
        /// <summary>
        /// Gets the backend for a specific platform.
        /// </summary>
        public OnnxBackend GetBackendForPlatform(RuntimePlatform platform)
        {
            foreach (var ov in _platformOverrides)
            {
                if (ov.Platform == platform && ov.Backend != null)
                    return ov.Backend;
            }
            
            return _defaultBackend;
        }
        
        /// <summary>
        /// Loads settings from Resources.
        /// </summary>
        public static OnnxSettings Load()
        {
            return Resources.Load<OnnxSettings>("OnnxSettings");
        }
    }
}
