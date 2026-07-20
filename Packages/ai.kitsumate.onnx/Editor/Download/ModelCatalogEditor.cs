using System;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Editor.Download
{
    [CustomEditor(typeof(ModelCatalog))]
    internal sealed class ModelCatalogEditor : UnityEditor.Editor
    {
        private ModelSet targetModelSet;
        private int variant;
        private bool installing;
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            var catalog = (ModelCatalog)target;
            EditorGUILayout.Space();
            targetModelSet = (ModelSet)EditorGUILayout.ObjectField("Target Model Set", targetModelSet, typeof(ModelSet), false);
            string[] names = Array.ConvertAll(catalog.Variants ?? Array.Empty<ModelVariantManifest>(), x => $"{x.ModelId} {x.Variant}");
            if (names.Length > 0) variant = EditorGUILayout.Popup("Variant", Mathf.Clamp(variant, 0, names.Length - 1), names);
            using (new EditorGUI.DisabledScope(installing || targetModelSet == null || names.Length == 0))
                if (GUILayout.Button("Install Selected Variant")) _ = InstallAsync(catalog);
        }
        private async Awaitable InstallAsync(ModelCatalog catalog)
        {
            installing = true;
            try
            {
                OnnxSettings settings = OnnxSettings.Load();
                string root = settings != null ? settings.ModelStorageRoot : "Assets/StreamingAssets/KitsuMateModels";
                await ModelManifestInstaller.InstallIntoAsync(catalog.Variants[variant], targetModelSet, root);
            }
            catch (Exception exception) { Debug.LogException(exception); EditorUtility.DisplayDialog("Model installation failed", exception.Message, "OK"); }
            finally { installing = false; Repaint(); }
        }
    }
}
