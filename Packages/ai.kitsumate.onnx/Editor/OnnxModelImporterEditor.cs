#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace KitsuMate.Onnx.Editor
{
    /// <summary>
    /// Custom editor for OnnxModelImporter.
    /// Provides a lightweight inspector that doesn't cause serialization lag.
    /// </summary>
    [CustomEditor(typeof(OnnxModelImporter))]
    public class OnnxModelImporterEditor : ScriptedImporterEditor
    {
        public override bool RequiresConstantRepaint() => false;
        
        public override void OnInspectorGUI()
        {
            // Minimal importer UI - the asset editor shows full details
            EditorGUILayout.HelpBox(
                "This is an ONNX Runtime model file (.onnx or .ort). See the imported asset below for model details.",
                MessageType.Info);
            
            ApplyRevertGUI();
        }
        
        // Prevent the default property iteration which causes lag
        protected override bool needsApplyRevert => false;
    }
}
#endif
