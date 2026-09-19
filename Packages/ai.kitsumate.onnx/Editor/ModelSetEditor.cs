using System;
using System.Linq;
using KitsuMate.Onnx.Download;
using KitsuMate.Onnx.Editor.Download;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Editor
{
    [CustomEditor(typeof(ModelSet), true)]
    public class ModelSetEditor : UnityEditor.Editor
    {
        protected virtual bool UsesSentis => false;
        protected virtual void Installed(DownloadedModel installation) { }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            var set = (ModelSet)target;
            DrawPropertiesExcluding(serializedObject, "m_Script", "download");
            serializedObject.ApplyModifiedProperties();
            using (new EditorGUI.DisabledScope(targets.Length != 1))
            {
                if (GUILayout.Button("Download models...")) ModelDownloadWindow.Show(set, useSentis: UsesSentis);
                int files = set.GetAllModels().OfType<OnnxModelReference>().Count(source => source.Kind == OnnxModelReference.SourceKind.File) +
                    set.GetAllTextFiles().Count(source => source != null && source.Kind == OnnxModelReference.SourceKind.File);
                EditorGUILayout.LabelField("File references", files.ToString());
                using (new EditorGUI.DisabledScope(files == 0 && !UsesSentis))
                    if (GUILayout.Button("Move files into Assets"))
                    {
                        try
                        {
                            EnsureCanMove(set);
                            if (UsesSentis) Installed(set.Installation(OnnxSettings.Load().InstallationRoot).Read()
                                ?? ImportedModelGraphs.MigratedInstallation(set)
                                ?? throw new InvalidOperationException("Download models before importing Unity AI Inference assets."));
                            ImportedModelGraphs.MoveAssignedFiles(set);
                        }
                        catch (Exception exception) { Debug.LogException(exception); }
                    }
                int assets = set.GetAllModels().OfType<OnnxModelReference>().Count(source => source.Kind == OnnxModelReference.SourceKind.Asset && source.Asset != null) +
                    set.GetAllTextFiles().Count(source => source != null && source.Kind == OnnxModelReference.SourceKind.Asset && source.Asset != null);
                using (new EditorGUI.DisabledScope(assets == 0))
                    if (GUILayout.Button(new GUIContent("Move files into data folder", "Move sources to Application.persistentDataPath and use File references.")))
                    {
                        try { EnsureCanMove(set); ImportedModelGraphs.MoveAssignedFilesToData(set); }
                        catch (Exception exception) { Debug.LogException(exception); }
                    }
                if (UsesSentis) EditorGUILayout.LabelField("Unity AI Inference graphs must remain imported assets.", EditorStyles.wordWrappedMiniLabel);
            }
        }

        private static void EnsureCanMove(ModelSet set)
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
                throw new InvalidOperationException("Stop Play Mode before moving model files.");
            if (InferenceEngineRuntimeBase.LiveRuntimes.Any(runtime => runtime.SourceModelSet == set))
                throw new InvalidOperationException("Unload previews and other runtimes using this model set before moving its files.");
        }
    }
}
