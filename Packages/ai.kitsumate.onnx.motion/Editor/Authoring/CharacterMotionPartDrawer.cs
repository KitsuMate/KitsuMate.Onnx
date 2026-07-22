using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Editor
{
    [CustomPropertyDrawer(typeof(CharacterMotionPart))]
    internal sealed class CharacterMotionPartDrawer : PropertyDrawer
    {
        private const float Gap = 2f;

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            EditorGUI.BeginProperty(position, label, property);
            SerializedProperty intent = property.FindPropertyRelative("intent");
            SerializedProperty repetitions = property.FindPropertyRelative("repetitions");
            SerializedProperty overrideSettings = property.FindPropertyRelative("overrideGenerationSettings");
            SerializedProperty settings = property.FindPropertyRelative("generationSettings");
            float line = EditorGUIUtility.singleLineHeight;

            var foldoutRect = new Rect(position.x, position.y, position.width, line);
            property.isExpanded = EditorGUI.Foldout(foldoutRect, property.isExpanded, label, true);
            if (!property.isExpanded)
            {
                EditorGUI.EndProperty();
                return;
            }

            EditorGUI.indentLevel++;
            float y = position.y + line + Gap;
            EditorGUI.PropertyField(new Rect(position.x, y, position.width, line), intent);
            y += line + Gap;
            EditorGUI.PropertyField(new Rect(position.x, y, position.width, line), repetitions);
            y += line + Gap;
            EditorGUI.PropertyField(new Rect(position.x, y, position.width, line), overrideSettings,
                new GUIContent("Override Generation Settings"));
            if (overrideSettings.boolValue)
            {
                y += line + Gap;
                float height = EditorGUI.GetPropertyHeight(settings, true);
                EditorGUI.PropertyField(new Rect(position.x, y, position.width, height), settings, true);
            }
            EditorGUI.indentLevel--;
            EditorGUI.EndProperty();
        }

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            float line = EditorGUIUtility.singleLineHeight;
            if (!property.isExpanded) return line;
            float result = line * 4f + Gap * 3f;
            SerializedProperty overrideSettings = property.FindPropertyRelative("overrideGenerationSettings");
            if (overrideSettings.boolValue)
            {
                SerializedProperty settings = property.FindPropertyRelative("generationSettings");
                result += Gap + EditorGUI.GetPropertyHeight(settings, true);
            }
            return result;
        }
    }
}
