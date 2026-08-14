#if UNITY_EDITOR
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.LipSync.Uni2005;
using KitsuMate.Onnx.Editor.Download;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.LipSync.Editor
{
    [CustomEditor(typeof(Uni2005ModelSet))]
    public sealed class Uni2005ModelSetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update(); var set = (Uni2005ModelSet)target;
            ModelSetEditorUi.Header(set, "Uni2005 Lip Sync Model Set");
            EditorGUILayout.LabelField("ONNX Models", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_acousticModelSource"), new GUIContent("Acoustic Model"));
            EditorGUILayout.LabelField("Auxiliary Files", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_vocabulary"));
            serializedObject.ApplyModifiedProperties();
            if (GUILayout.Button("Download Models"))
                ShowDownload(set);
            ModelSetEditorUi.Validation(set);
        }

        internal static void ShowDownload(Uni2005ModelSet set)
        {
            ModelDownloadWindow.Show(new ModelDownloadRequest("KitsuMate/uni2005-onnx",
                "6218cd475c82f60ae6910832f1a713936a327006", "uni2005"), result =>
            {
                result.RequireGraph("model", new[] { "mfcc" });
                result.ConfigureModel(set.AcousticModel, "model");
                set.SetVocabulary(result.LoadAsset<TextAsset>("vocabulary"));
                result.ApplyMetadata(set);
                AssetDatabase.SaveAssets();
                Selection.activeObject = set;
            });
        }

    }

    [CustomEditor(typeof(Uni2005Engine))]
    public sealed class Uni2005EngineEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            SerializedProperty modelSet = serializedObject.FindProperty("modelSet");
            if (modelSet.objectReferenceValue != null || !GUILayout.Button("Create Model Set and Download")) return;
            string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(target))?.Replace('\\', '/') ?? "Assets";
            var set = CreateInstance<Uni2005ModelSet>();
            AssetDatabase.CreateAsset(set, AssetDatabase.GenerateUniqueAssetPath($"{directory}/Uni2005ModelSet.asset"));
            modelSet.objectReferenceValue = set;
            serializedObject.ApplyModifiedProperties();
            OnnxSettings settings = OnnxSettings.Load();
            if (settings != null) settings.RegisterDefaultEngine((Uni2005Engine)target);
            AssetDatabase.SaveAssets();
            Uni2005ModelSetEditor.ShowDownload(set);
        }
    }
}
#endif
