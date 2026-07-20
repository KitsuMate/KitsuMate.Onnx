#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace KitsuMate.Onnx.Editor
{
    [CustomPropertyDrawer(typeof(OnnxModelReference))]
    public sealed class OnnxModelReferenceDrawer : PropertyDrawer
    {
        private sealed class TestState { public GpuProvider Provider; public bool Running; public string Message; public MessageType Type; }
        private static readonly Dictionary<string, TestState> States = new();

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            var state = GetState(property);
            int lines = 4 + (string.IsNullOrWhiteSpace(state.Message) ? 0 : 2);
            return lines * (EditorGUIUtility.singleLineHeight + EditorGUIUtility.standardVerticalSpacing) + 4;
        }

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            EditorGUI.BeginProperty(position, label, property);
            float h = EditorGUIUtility.singleLineHeight, gap = EditorGUIUtility.standardVerticalSpacing;
            Rect line = new(position.x, position.y, position.width, h);
            var kind = property.FindPropertyRelative("sourceKind");
            var asset = property.FindPropertyRelative("asset");
            var path = property.FindPropertyRelative("relativePath");
            EditorGUI.BeginChangeCheck();
            EditorGUI.PropertyField(line, kind, label);
            bool kindChanged = EditorGUI.EndChangeCheck();
            line.y += h + gap;
            if ((OnnxModelReference.SourceKind)kind.enumValueIndex == OnnxModelReference.SourceKind.Asset)
                EditorGUI.PropertyField(line, asset, new GUIContent("ONNX Asset"));
            else
            {
                EditorGUI.BeginChangeCheck();
                EditorGUI.PropertyField(line, path, new GUIContent("Model Path", "Relative to StreamingAssets/KitsuMateModels."));
                if (EditorGUI.EndChangeCheck()) ClearFileMetadata(property);
            }
            if (kindChanged) ClearFileMetadata(property);

            property.serializedObject.ApplyModifiedProperties();
            var reference = GetReference(property);
            line.y += h + gap;
            string availability = reference == null ? "Unassigned" : reference.IsAvailable ? (reference.IsMetadataStale ? "Stale" : "Ready") : "Missing";
            EditorGUI.LabelField(line, "Status", reference == null ? availability : $"{availability}  |  {reference.SourceName}");
            line.y += h + gap;
            var state = GetState(property);
            float third = (line.width - 8) / 3f;
            state.Provider = (GpuProvider)EditorGUI.EnumPopup(new Rect(line.x, line.y, third, h), state.Provider);
            using (new EditorGUI.DisabledScope(state.Running || reference?.Kind != OnnxModelReference.SourceKind.File || reference?.IsAvailable != true))
                if (GUI.Button(new Rect(line.x + third + 4, line.y, third, h), "Refresh Metadata")) RefreshAsync(property, reference, state);
            using (new EditorGUI.DisabledScope(state.Running || reference?.IsAvailable != true))
                if (GUI.Button(new Rect(line.x + (third + 4) * 2, line.y, third, h), state.Running ? "Working..." : "Test Session"))
                    TestAsync(property, reference, state);
            if (!string.IsNullOrWhiteSpace(state.Message))
            {
                line.y += h + gap;
                EditorGUI.HelpBox(new Rect(line.x, line.y, line.width, h * 2), state.Message, state.Type);
            }
            EditorGUI.EndProperty();
        }

        private static string Key(SerializedProperty p) => $"{p.serializedObject.targetObject.GetEntityId()}:{p.propertyPath}";
        private static void ClearFileMetadata(SerializedProperty property)
        {
            property.FindPropertyRelative("sha256").stringValue = string.Empty;
            property.FindPropertyRelative("cachedFileSize").longValue = 0;
            property.FindPropertyRelative("cachedWriteTimeUtcTicks").longValue = 0;
            property.FindPropertyRelative("metadataInspected").boolValue = false;
            property.FindPropertyRelative("inputs").arraySize = 0;
            property.FindPropertyRelative("outputs").arraySize = 0;
        }
        private static TestState GetState(SerializedProperty p) { string key = Key(p); if (!States.TryGetValue(key, out var s)) States[key] = s = new TestState(); return s; }
        private OnnxModelReference GetReference(SerializedProperty property) => fieldInfo?.GetValue(property.serializedObject.targetObject) as OnnxModelReference;

        private static async void TestAsync(SerializedProperty property, OnnxModelReference reference, TestState state)
        {
            state.Running = true; state.Message = $"Creating {state.Provider} session in the background..."; state.Type = MessageType.Info;
            string path = reference.ResolveModelPath();
            byte[] data = string.IsNullOrWhiteSpace(path) ? reference.ImportedAsset?.CopyModelData() : null;
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            backend.EnableGpu = state.Provider != GpuProvider.CPU; backend.PreferredProvider = state.Provider;
            try
            {
                state.Message = await Task.Run(() => { var sw = Stopwatch.StartNew(); using IOnnxSession session = data != null ? backend.CreateSession(data) : backend.CreateSession(path); return $"Created in {sw.ElapsedMilliseconds} ms. {session.InputNames.Count} inputs, {session.OutputNames.Count} outputs."; });
                state.Type = MessageType.Info;
            }
            catch (Exception e) { state.Message = e.Message; state.Type = MessageType.Error; Debug.LogException(e, property.serializedObject.targetObject); }
            finally { backend.Dispose(); UnityEngine.Object.DestroyImmediate(backend); state.Running = false; InternalEditorUtility.RepaintAllViews(); }
        }

        private static async void RefreshAsync(SerializedProperty property, OnnxModelReference reference, TestState state)
        {
            state.Running = true; state.Message = "Inspecting model metadata in the background..."; state.Type = MessageType.Info;
            string path = reference.ResolveModelPath(); string relative = reference.RelativePath;
            string propertyPath = property.propertyPath;
            UnityEngine.Object owner = property.serializedObject.targetObject;
            try
            {
                var result = await Task.Run(() => ExternalOnnxModelAssetUtility.Inspect(path));
                if (owner == null) return;
                Undo.RecordObject(owner, "Refresh ONNX Metadata");
                var serialized = new SerializedObject(owner);
                SerializedProperty refreshed = serialized.FindProperty(propertyPath);
                if (refreshed == null) throw new InvalidOperationException($"ONNX model property '{propertyPath}' no longer exists.");
                refreshed.FindPropertyRelative("sourceKind").enumValueIndex = (int)OnnxModelReference.SourceKind.File;
                refreshed.FindPropertyRelative("relativePath").stringValue = relative;
                refreshed.FindPropertyRelative("metadataInspected").boolValue = result.Outputs.Length > 0;
                var file = new FileInfo(path);
                refreshed.FindPropertyRelative("cachedFileSize").longValue = file.Length;
                refreshed.FindPropertyRelative("cachedWriteTimeUtcTicks").longValue = file.LastWriteTimeUtc.Ticks;
                SetTensorInfos(refreshed.FindPropertyRelative("inputs"), result.Inputs);
                SetTensorInfos(refreshed.FindPropertyRelative("outputs"), result.Outputs);
                serialized.ApplyModifiedProperties();
                EditorUtility.SetDirty(owner);
                state.Message = $"Metadata refreshed: {result.Inputs.Length} inputs, {result.Outputs.Length} outputs.";
                state.Type = MessageType.Info;
            }
            catch (Exception e) { state.Message = e.Message; state.Type = MessageType.Error; }
            finally { state.Running = false; InternalEditorUtility.RepaintAllViews(); }
        }

        private static void SetTensorInfos(SerializedProperty target, OnnxModelAsset.TensorInfo[] values)
        {
            values ??= Array.Empty<OnnxModelAsset.TensorInfo>(); target.arraySize = values.Length;
            for (int i = 0; i < values.Length; i++)
            {
                SerializedProperty item = target.GetArrayElementAtIndex(i);
                item.FindPropertyRelative("name").stringValue = values[i].Name ?? string.Empty;
                item.FindPropertyRelative("elementType").stringValue = values[i].ElementType ?? string.Empty;
                item.FindPropertyRelative("shapeDescription").stringValue = values[i].ShapeDescription ?? string.Empty;
                int[] dimensions = values[i].Shape ?? Array.Empty<int>();
                SerializedProperty shape = item.FindPropertyRelative("shape"); shape.arraySize = dimensions.Length;
                for (int dimension = 0; dimension < dimensions.Length; dimension++) shape.GetArrayElementAtIndex(dimension).intValue = dimensions[dimension];
            }
        }
    }

    public static class ModelSetEditorUi
    {
        public static void Header(ModelSet modelSet, string title)
        {
            EditorGUILayout.LabelField(title, EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(modelSet.IsComplete ? $"{modelSet.DisplayName} is ready." : "Model set is incomplete.", modelSet.IsComplete ? MessageType.Info : MessageType.Warning);
        }

        public static void Validation(ModelSet modelSet)
        {
            ModelValidationResult result = modelSet.Validate(new ModelValidationContext(null));
            foreach (ModelDiagnostic diagnostic in result.Diagnostics)
                EditorGUILayout.HelpBox(diagnostic.Message, diagnostic.Severity == ModelDiagnosticSeverity.Error ? MessageType.Error : MessageType.Warning);
            EditorGUILayout.HelpBox("Install or update variants through the shared ONNX Model Catalog.", MessageType.Info);
        }
    }
}
#endif
