#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Editor
{
    /// <summary>
    /// Shared inspector presentation for feature model sets.
    /// </summary>
    public static class ModelSetEditorUi
    {
        public static void Header(ModelSet modelSet, string title)
        {
            EditorGUILayout.LabelField(title, EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                modelSet.IsComplete ? $"{modelSet.DisplayName} is ready." : "Model set is incomplete.",
                modelSet.IsComplete ? MessageType.Info : MessageType.Warning);
        }

        public static void Validation(ModelSet modelSet)
        {
            ModelValidationResult result = modelSet.Validate(new ModelValidationContext(null));
            foreach (ModelDiagnostic diagnostic in result.Diagnostics)
            {
                EditorGUILayout.HelpBox(
                    diagnostic.Message,
                    diagnostic.Severity == ModelDiagnosticSeverity.Error ? MessageType.Error : MessageType.Warning);
            }

            EditorGUILayout.HelpBox("Use Download Models in this inspector to install a compatible verified variant.", MessageType.Info);
        }
    }
}
#endif
