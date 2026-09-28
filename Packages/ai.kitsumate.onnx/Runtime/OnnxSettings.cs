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
        internal const string DefaultInstallationFolder = "KitsuMateModels";

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
        [SerializeField] private string _installationFolder = DefaultInstallationFolder;
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
        /// <summary>Environment variable that points every project and build on a machine to one data folder.</summary>
        public const string DataRootVariable = "KITSUMATE_ONNX_DATA_DIR";
        private static string _dataRoot;

        /// <summary>
        /// Folder that holds installed models and data-folder file references. Uses <see cref="DataRootVariable"/>
        /// when set, then the value set by the application, then <see cref="Application.persistentDataPath"/>.
        /// </summary>
        public static string DataRoot
        {
            get
            {
                string shared = Environment.GetEnvironmentVariable(DataRootVariable);
                if (!string.IsNullOrWhiteSpace(shared)) return shared;
                return string.IsNullOrWhiteSpace(_dataRoot) ? Application.persistentDataPath : _dataRoot;
            }
            set => _dataRoot = value;
        }

        public string InstallationRoot => Download.ModelDownloadPaths.Child(DataRoot, _installationFolder);
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
