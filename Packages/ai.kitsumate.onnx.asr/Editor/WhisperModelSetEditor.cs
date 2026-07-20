#if UNITY_EDITOR
using KitsuMate.Onnx.Asr.Whisper;
using KitsuMate.Onnx.Editor;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Asr.Editor
{
    [CustomEditor(typeof(WhisperModelSet))]
    public sealed class WhisperModelSetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update(); var set = (WhisperModelSet)target;
            ModelSetEditorUi.Header(set, "Whisper Model Set");
            EditorGUILayout.LabelField("ONNX Models", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_melProcessorSource"), new GUIContent("Mel Processor"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_encoderSource"), new GUIContent("Encoder"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_decoderSource"), new GUIContent("Decoder"));
            EditorGUILayout.LabelField("Auxiliary Files", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_tokenizerJson"), new GUIContent("Tokenizer JSON"));
            serializedObject.ApplyModifiedProperties(); ModelSetEditorUi.Validation(set);
        }
    }
}
#endif
