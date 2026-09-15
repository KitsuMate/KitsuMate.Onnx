using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Editor
{
    [CustomPropertyDrawer(typeof(TextFileReference))]
    public sealed class TextFileReferenceDrawer : PropertyDrawer
    {
        public override float GetPropertyHeight(SerializedProperty property, GUIContent label) =>
            (EditorGUIUtility.singleLineHeight + 2) * (property.FindPropertyRelative("sourceKind").enumValueIndex == 0 ? 2 : 3);

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            EditorGUI.BeginProperty(position, label, property);
            position.height = EditorGUIUtility.singleLineHeight;
            var kind = property.FindPropertyRelative("sourceKind");
            EditorGUI.PropertyField(position, kind, label);
            position.y += position.height + 2;
            if (kind.enumValueIndex == 0) EditorGUI.PropertyField(position, property.FindPropertyRelative("asset"), new GUIContent("Asset"));
            else
            {
                EditorGUI.PropertyField(position, property.FindPropertyRelative("fileRoot"), new GUIContent("Relative to"));
                position.y += position.height + 2;
                var path = property.FindPropertyRelative("filePath");
                var browse = new Rect(position.xMax - 65, position.y, 65, position.height);
                position.width -= 69;
                EditorGUI.PropertyField(position, path, new GUIContent("File"));
                if (GUI.Button(browse, "Browse"))
                {
                    string selected = EditorUtility.OpenFilePanel("Select text file", "", "");
                    if (!string.IsNullOrEmpty(selected))
                    {
                        var reference = TextFileReference.FromFile(selected);
                        property.boxedValue = reference;
                    }
                }
            }
            EditorGUI.EndProperty();
        }
    }
}
