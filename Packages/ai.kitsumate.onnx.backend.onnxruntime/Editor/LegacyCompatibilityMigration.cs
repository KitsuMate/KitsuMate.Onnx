#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Editor
{
    /// <summary>
    /// Opt-in conversion for assets authored before the backend-neutral
    /// compatibility contract. This deliberately runs only from the menu so a
    /// package upgrade never rewrites project assets implicitly.
    /// </summary>
    internal static class LegacyCompatibilityMigration
    {
        private const string BackendId = "onnxruntime";

        [MenuItem("KitsuMate/ONNX/Migration/Convert legacy ONNX Runtime compatibility")]
        private static void ConvertLegacyCompatibility()
        {
            if (!EditorUtility.DisplayDialog(
                    "Convert ONNX Runtime compatibility",
                    "This updates project ModelSet assets that have legacy provider entries to use the backend-neutral 'onnxruntime' identifier. Review the changed assets before committing.",
                    "Convert", "Cancel"))
                return;

            int assetCount = 0;
            int entryCount = 0;
            foreach (string guid in AssetDatabase.FindAssets("t:ScriptableObject", new[] { "Assets" }))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                var modelSet = AssetDatabase.LoadAssetAtPath<ModelSet>(path);
                if (modelSet == null) continue;
                var serialized = new SerializedObject(modelSet);
                SerializedProperty compatibility = serialized.FindProperty("providerCompatibility");
                if (compatibility == null) continue;

                bool changed = false;
                for (int index = 0; index < compatibility.arraySize; index++)
                {
                    SerializedProperty entry = compatibility.GetArrayElementAtIndex(index);
                    SerializedProperty backendId = entry.FindPropertyRelative("BackendId");
                    if (backendId == null || !string.IsNullOrWhiteSpace(backendId.stringValue)) continue;
                    backendId.stringValue = BackendId;
                    changed = true;
                    entryCount++;
                }

                if (!changed) continue;
                serialized.ApplyModifiedProperties();
                EditorUtility.SetDirty(modelSet);
                assetCount++;
            }

            if (assetCount > 0) AssetDatabase.SaveAssets();
            Debug.Log($"[KitsuMate ONNX] Converted {entryCount} legacy compatibility entries across {assetCount} ModelSet assets. Legacy provider selection is preserved as metadata only; review model/provider requirements before shipping.");
        }
    }
}
#endif
