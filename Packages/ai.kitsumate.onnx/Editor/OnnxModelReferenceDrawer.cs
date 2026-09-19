#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Threading;
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
        private sealed class TestState { public bool Running; public string Message; public MessageType Type; public CancellationTokenSource Cancellation; }
        private static readonly Dictionary<string, TestState> States = new();

        static OnnxModelReferenceDrawer()
        {
            AssemblyReloadEvents.beforeAssemblyReload += () =>
            {
                foreach (TestState state in States.Values) state.Cancellation?.Cancel();
                States.Clear();
            };
        }

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            var state = GetState(property);
            int lines = 3 + (property.FindPropertyRelative("sourceKind").enumValueIndex == (int)OnnxModelReference.SourceKind.File ? 1 : 0)
                + (string.IsNullOrWhiteSpace(state.Message) ? 0 : 2);
            return lines * (EditorGUIUtility.singleLineHeight + EditorGUIUtility.standardVerticalSpacing) + 4;
        }

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            EditorGUI.BeginProperty(position, label, property);
            float h = EditorGUIUtility.singleLineHeight, gap = EditorGUIUtility.standardVerticalSpacing;
            Rect line = new(position.x, position.y, position.width, h);
            var kind = property.FindPropertyRelative("sourceKind");
            var asset = property.FindPropertyRelative("asset");
            var path = property.FindPropertyRelative("filePath");
            EditorGUI.BeginChangeCheck();
            EditorGUI.PropertyField(line, kind, label);
            bool kindChanged = EditorGUI.EndChangeCheck();
            line.y += h + gap;
            if ((OnnxModelReference.SourceKind)kind.enumValueIndex == OnnxModelReference.SourceKind.Asset)
                EditorGUI.PropertyField(line, asset, new GUIContent("ONNX Asset"));
            else
            {
                EditorGUI.BeginChangeCheck();
                EditorGUI.PropertyField(line, property.FindPropertyRelative("fileRoot"), new GUIContent("Relative to"));
                line.y += h + gap;
                EditorGUI.PropertyField(line, path, new GUIContent("Model path", "Persistent Data paths are relative to Application.persistentDataPath. Absolute paths are specific to this machine."));
                if (EditorGUI.EndChangeCheck()) ClearFileMetadata(property);
            }
            if (kindChanged) ClearFileMetadata(property);

            property.serializedObject.ApplyModifiedProperties();
            var reference = GetReference(property);
            line.y += h + gap;
            string availability = reference == null ? "Unassigned" : reference.IsAvailable ? (reference.IsMetadataStale ? "Stale" : "Ready") : "Missing";
            EditorGUI.LabelField(line, "Status", reference == null ? availability : $"{availability}  |  {reference.SourceName}");
            var state = GetState(property);
            if (reference?.Kind == OnnxModelReference.SourceKind.File && reference.IsAvailable)
            {
                var button = new Rect(line.xMax - 24, line.y, 24, h);
                using (new EditorGUI.DisabledScope(state.Running))
                    if (GUI.Button(button, "…", EditorStyles.miniButton))
                    {
                        var menu = new GenericMenu();
                        menu.AddItem(new GUIContent("Refresh metadata"), false, () => RefreshAsync(property, reference, state));
                        menu.AddItem(new GUIContent("Copy resolved path"), false, () => EditorGUIUtility.systemCopyBuffer = reference.ResolveModelPath());
                        menu.ShowAsContext();
                    }
            }
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

        private static async void RefreshAsync(SerializedProperty property, OnnxModelReference reference, TestState state)
        {
            state.Cancellation?.Cancel();
            state.Cancellation?.Dispose();
            state.Cancellation = new CancellationTokenSource();
            CancellationToken cancellationToken = state.Cancellation.Token;
            state.Running = true; state.Message = "Inspecting model metadata in the background..."; state.Type = MessageType.Info;
            string path = reference.ResolveModelPath(); string relative = reference.FilePath;
            string propertyPath = property.propertyPath;
            UnityEngine.Object owner = property.serializedObject.targetObject;
            try
            {
                var result = await Task.Run(() => { cancellationToken.ThrowIfCancellationRequested(); return OnnxLightweightMetadataReader.Read(path); }, cancellationToken);
                cancellationToken.ThrowIfCancellationRequested();
                if (owner == null) return;
                Undo.RecordObject(owner, "Refresh ONNX Metadata");
                var serialized = new SerializedObject(owner);
                SerializedProperty refreshed = serialized.FindProperty(propertyPath);
                if (refreshed == null) throw new InvalidOperationException($"ONNX model property '{propertyPath}' no longer exists.");
                refreshed.FindPropertyRelative("sourceKind").enumValueIndex = (int)OnnxModelReference.SourceKind.File;
                refreshed.FindPropertyRelative("filePath").stringValue = relative;
                refreshed.FindPropertyRelative("metadataInspected").boolValue = result.Outputs.Count > 0;
                var file = new FileInfo(path);
                refreshed.FindPropertyRelative("cachedFileSize").longValue = file.Length;
                refreshed.FindPropertyRelative("cachedWriteTimeUtcTicks").longValue = file.LastWriteTimeUtc.Ticks;
                SetTensorInfos(refreshed.FindPropertyRelative("inputs"), result.Inputs.ToArray());
                SetTensorInfos(refreshed.FindPropertyRelative("outputs"), result.Outputs.ToArray());
                serialized.ApplyModifiedProperties();
                EditorUtility.SetDirty(owner);
                state.Message = $"Metadata refreshed: {result.Inputs.Count} inputs, {result.Outputs.Count} outputs.";
                state.Type = MessageType.Info;
            }
            catch (OperationCanceledException) { state.Message = "Metadata refresh cancelled."; state.Type = MessageType.Warning; }
            catch (Exception e) { state.Message = e.Message; state.Type = MessageType.Error; }
            finally { state.Running = false; state.Cancellation?.Dispose(); state.Cancellation = null; InternalEditorUtility.RepaintAllViews(); }
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

}
#endif
