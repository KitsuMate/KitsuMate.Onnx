#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Editor
{
    /// <summary>
    /// Custom inspector for OnnxModelAsset.
    /// Shows model metadata, inputs/outputs, and provides testing tools.
    /// </summary>
    [CustomEditor(typeof(OnnxModelAsset))]
    public class OnnxModelAssetEditor : UnityEditor.Editor
    {
        private bool _showInputs = true;
        private bool _showOutputs = true;
        private bool _showMetadata;
        
        // Cache to avoid repeated property lookups
        private OnnxModelAsset _cachedAsset;
        private string _cachedModelName;
        private string _cachedProducer;
        private string _cachedOpsetVersion;
        private string _cachedOriginalFileSize;
        private string _cachedEmbeddedSize;
        private string _cachedImportedUtc;
        private string _cachedDescription;
        private int _cachedInputCount;
        private int _cachedOutputCount;
        private int _cachedMetadataCount;
        private bool _cachedHasModelData;
        private bool _cachedHasResolvableData;
        private OnnxModelAsset.MetadataInspectionState _cachedMetadataState;
        private string _cachedMetadataError;
        private OnnxOptimizationLevel _cachedOptLevel;
        private bool _cachedExternalDataMerged;
        
        private void CacheAssetData(OnnxModelAsset asset)
        {
            if (_cachedAsset == asset && _cachedOptLevel == asset.DefaultOptimizationLevel)
                return;
                
            _cachedAsset = asset;
            _cachedModelName = asset.ModelName;
            _cachedProducer = string.IsNullOrEmpty(asset.Producer) ? "—" : asset.Producer;
            _cachedOpsetVersion = asset.OpsetVersion > 0 ? asset.OpsetVersion.ToString() : "—";
            _cachedOriginalFileSize = asset.FileSizeFormatted;
            _cachedEmbeddedSize = asset.EmbeddedSizeFormatted;
            _cachedImportedUtc = asset.ImportedUtc;
            _cachedDescription = asset.Description;
            _cachedInputCount = asset.Inputs.Count;
            _cachedOutputCount = asset.Outputs.Count;
            _cachedMetadataCount = asset.CustomMetadata.Count;
            _cachedHasModelData = asset.HasModelData;
            _cachedHasResolvableData = asset.HasResolvableData;
            _cachedMetadataState = asset.MetadataState;
            _cachedMetadataError = asset.MetadataError;
            _cachedOptLevel = asset.DefaultOptimizationLevel;
            _cachedExternalDataMerged = asset.ExternalDataMerged;
        }
        
        public override bool RequiresConstantRepaint() => false;
        
        // Prevent serializedObject access which causes lag with large byte arrays
        protected override bool ShouldHideOpenButton() => false;
        
        public override void OnInspectorGUI()
        {
            // Do NOT call serializedObject.Update() - it causes lag with large model data
            var asset = (OnnxModelAsset)target;
            
            // Cache data to avoid repeated serialized property access
            CacheAssetData(asset);
            
            // Enable GUI - imported assets are read-only by default
            var previousEnabled = GUI.enabled;
            GUI.enabled = true;
            
            // Header
            EditorGUILayout.LabelField("ONNX Model Asset", EditorStyles.boldLabel);
            EditorGUILayout.Space(4);
            
            // Status
            DrawStatusBox();
            EditorGUILayout.Space(8);
            
            // Model Info
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                EditorGUILayout.LabelField("Model Information", EditorStyles.boldLabel);
                EditorGUI.indentLevel++;
                
                EditorGUILayout.LabelField("Name", _cachedModelName);
                EditorGUILayout.LabelField("Producer", _cachedProducer);
                EditorGUILayout.LabelField("Opset Version", _cachedOpsetVersion);
                
                EditorGUILayout.Space(4);
                
                if (_cachedExternalDataMerged)
                {
                    EditorGUILayout.LabelField("Original File Size", _cachedOriginalFileSize);
                    EditorGUILayout.LabelField("Embedded Size", $"{_cachedEmbeddedSize} (merged from external data)");
                }
                else
                {
                    EditorGUILayout.LabelField("Size", _cachedEmbeddedSize);
                }
                
                EditorGUILayout.LabelField("Imported", _cachedImportedUtc);
                
                if (!string.IsNullOrEmpty(_cachedDescription))
                {
                    EditorGUILayout.Space(4);
                    EditorGUILayout.LabelField("Description", EditorStyles.miniLabel);
                    EditorGUILayout.HelpBox(_cachedDescription, MessageType.None);
                }
                
                EditorGUI.indentLevel--;
            }
            
            EditorGUILayout.Space(4);
            
            // Inputs
            _showInputs = EditorGUILayout.BeginFoldoutHeaderGroup(_showInputs, $"Inputs ({_cachedInputCount})");
            if (_showInputs)
            {
                using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
                {
                    foreach (var input in asset.Inputs)
                    {
                        EditorGUILayout.LabelField(input.Name, $"{input.ElementType}[{input.ShapeDescription}]");
                    }
                    if (_cachedInputCount == 0)
                    {
                        EditorGUILayout.LabelField("(none)", EditorStyles.miniLabel);
                    }
                }
            }
            EditorGUILayout.EndFoldoutHeaderGroup();
            
            // Outputs
            _showOutputs = EditorGUILayout.BeginFoldoutHeaderGroup(_showOutputs, $"Outputs ({_cachedOutputCount})");
            if (_showOutputs)
            {
                using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
                {
                    foreach (var output in asset.Outputs)
                    {
                        EditorGUILayout.LabelField(output.Name, $"{output.ElementType}[{output.ShapeDescription}]");
                    }
                    if (_cachedOutputCount == 0)
                    {
                        EditorGUILayout.LabelField("(none)", EditorStyles.miniLabel);
                    }
                }
            }
            EditorGUILayout.EndFoldoutHeaderGroup();
            
            // Custom Metadata
            if (_cachedMetadataCount > 0)
            {
                _showMetadata = EditorGUILayout.BeginFoldoutHeaderGroup(_showMetadata, $"Custom Metadata ({_cachedMetadataCount})");
                if (_showMetadata)
                {
                    using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
                    {
                        foreach (var entry in asset.CustomMetadata)
                        {
                            EditorGUILayout.LabelField(entry.Key, entry.Value);
                        }
                    }
                }
                EditorGUILayout.EndFoldoutHeaderGroup();
            }
            
            EditorGUILayout.Space(8);
            
            // Default session intent
            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                EditorGUILayout.LabelField("Default Session Intent", EditorStyles.boldLabel);
                EditorGUI.indentLevel++;
                
                var optLevel = (OnnxOptimizationLevel)EditorGUILayout.EnumPopup(
                    "Optimization Level", 
                    _cachedOptLevel);
                
                if (optLevel != _cachedOptLevel)
                {
                    Undo.RecordObject(asset, "Change Optimization Level");
                    asset.DefaultOptimizationLevel = optLevel;
                    _cachedOptLevel = optLevel;
                    EditorUtility.SetDirty(asset);
                }
                
                EditorGUI.indentLevel--;
            }
            
            // Restore previous GUI state
            GUI.enabled = previousEnabled;
        }
        
        private void DrawStatusBox()
        {
            var state = _cachedMetadataState;
            var (message, type) = state switch
            {
                OnnxModelAsset.MetadataInspectionState.Succeeded => ("Model imported successfully", MessageType.Info),
                OnnxModelAsset.MetadataInspectionState.Failed => ($"Metadata extraction failed: {_cachedMetadataError}", MessageType.Warning),
                _ => ("Metadata not inspected", MessageType.None)
            };
            
            if (!_cachedHasResolvableData)
            {
                message = "No resolvable model data";
                type = MessageType.Error;
            }
            else if (!_cachedHasModelData)
            {
                message = "External model file is available";
                type = MessageType.Info;
            }
            
            EditorGUILayout.HelpBox(message, type);
        }
        
    }
}
#endif
