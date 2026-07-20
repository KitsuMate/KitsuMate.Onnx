using System;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Editor
{
    [CustomEditor(typeof(CharacterMotionIntent))]
    internal sealed class CharacterMotionIntentEditor : UnityEditor.Editor
    {
        private SerializedProperty id;
        private SerializedProperty description;
        private SerializedProperty prompt;
        private SerializedProperty useCustomGenerationSettings;
        private SerializedProperty settings;
        private SerializedProperty embeddingEngine;

        private void OnEnable()
        {
            id = serializedObject.FindProperty("id");
            description = serializedObject.FindProperty("description");
            prompt = serializedObject.FindProperty("prompt");
            useCustomGenerationSettings = serializedObject.FindProperty("useCustomGenerationSettings");
            settings = serializedObject.FindProperty("settings");
            embeddingEngine = serializedObject.FindProperty("embeddingEngine");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EditorGUILayout.PropertyField(id);
            EditorGUILayout.PropertyField(description);
            EditorGUILayout.PropertyField(prompt);
            EditorGUILayout.PropertyField(useCustomGenerationSettings, new GUIContent("Custom Generation Settings"));
            if (useCustomGenerationSettings.boolValue)
                EditorGUILayout.PropertyField(settings, true);
            EditorGUILayout.Space();
            EditorGUILayout.PropertyField(embeddingEngine);
            serializedObject.ApplyModifiedProperties();

            var intent = (CharacterMotionIntent)target;
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Text Embedding", EditorStyles.boldLabel);

            if (intent.HasEmbedding)
            {
                bool current = intent.EmbeddingEngine != null && intent.EmbeddingEngine.ModelSet != null &&
                    intent.TryGetEmbedding(intent.EmbeddingEngine.ModelSet.Identity, out _);
                EditorGUILayout.HelpBox(
                    current ? "A current embedding is baked for the selected encoder." : "The baked embedding is stale or belongs to another encoder.",
                    current ? MessageType.Info : MessageType.Warning);
            }
            else
            {
                EditorGUILayout.HelpBox("No text embedding is baked.", MessageType.Info);
            }

            bool canBake = ValidateBakeConfiguration(intent);

            using (new EditorGUILayout.HorizontalScope())
            {
                using (new EditorGUI.DisabledScope(!canBake))
                {
                    if (GUILayout.Button(intent.HasEmbedding ? "Rebake Embedding" : "Bake Embedding"))
                        _ = RunAsync(intent);
                }

                using (new EditorGUI.DisabledScope(!intent.HasEmbedding))
                {
                    if (GUILayout.Button("Clear Embedding"))
                    {
                        Undo.RecordObject(intent, "Clear Character Motion Embedding");
                        intent.ClearEmbedding();
                        EditorUtility.SetDirty(intent);
                        AssetDatabase.SaveAssets();
                    }
                }
            }
        }

        private static bool ValidateBakeConfiguration(CharacterMotionIntent intent)
        {
            if (string.IsNullOrWhiteSpace(intent.Prompt))
            {
                EditorGUILayout.HelpBox("Enter a prompt before baking an embedding.", MessageType.Error);
                return false;
            }
            if (intent.EmbeddingEngine == null)
            {
                EditorGUILayout.HelpBox("Assign an embedding engine.", MessageType.Error);
                return false;
            }
            if (intent.EmbeddingEngine.ModelSet == null)
            {
                EditorGUILayout.HelpBox($"Embedding engine '{intent.EmbeddingEngine.name}' has no model set assigned.", MessageType.Error);
                return false;
            }
            if (intent.EmbeddingEngine.Backend == null)
            {
                EditorGUILayout.HelpBox($"Embedding engine '{intent.EmbeddingEngine.name}' has no backend assigned.", MessageType.Error);
                return false;
            }
            ModelValidationResult validation = intent.EmbeddingEngine.ModelSet.Validate(
                new ModelValidationContext(intent.EmbeddingEngine.Backend));
            foreach (ModelDiagnostic diagnostic in validation.Diagnostics.Where(value => value.Severity == ModelDiagnosticSeverity.Error))
                EditorGUILayout.HelpBox(diagnostic.Message, MessageType.Error);
            return validation.IsValid;
        }

        private static async Awaitable RunAsync(CharacterMotionIntent intent)
        {
            try { await CharacterMotionClipBaker.BakeEmbeddingAsync(intent); }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog("Character Motion Intent", exception.Message, "OK");
            }
        }
    }
}
