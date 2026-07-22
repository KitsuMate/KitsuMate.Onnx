using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Editor
{
    [CustomEditor(typeof(TextToSpeech))]
    public class TextToSpeechEditor : UnityEditor.Editor
    {
        private SerializedProperty _engine;
        private SerializedProperty _voiceReference;
        private SerializedProperty _languageId;
        private SerializedProperty _exaggeration;
        private SerializedProperty _maxNewTokens;
        private SerializedProperty _repetitionPenalty;
        private SerializedProperty _audioSource;
        private SerializedProperty _autoPlay;
        private SerializedProperty _onSynthesis;
        private SerializedProperty _onAudioClipReady;
        private SerializedProperty _onError;

        private string _previewText = "Hello, this is a test of the text to speech system.";
        private bool _isGenerating;
        private bool _showEvents;

        private void OnEnable()
        {
            _engine = serializedObject.FindProperty("_engine");
            _voiceReference = serializedObject.FindProperty("_voiceReference");
            _languageId = serializedObject.FindProperty("_languageId");
            _exaggeration = serializedObject.FindProperty("_exaggeration");
            _maxNewTokens = serializedObject.FindProperty("_maxNewTokens");
            _repetitionPenalty = serializedObject.FindProperty("_repetitionPenalty");
            _audioSource = serializedObject.FindProperty("_audioSource");
            _autoPlay = serializedObject.FindProperty("_autoPlay");
            _onSynthesis = serializedObject.FindProperty("_onSynthesis");
            _onAudioClipReady = serializedObject.FindProperty("_onAudioClipReady");
            _onError = serializedObject.FindProperty("_onError");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            var tts = (TextToSpeech)target;

            // Engine
            EditorGUILayout.LabelField("Engine", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(_engine);

            EditorGUILayout.Space(5);

            // Input
            EditorGUILayout.LabelField("Input", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(_voiceReference, new GUIContent("Voice Reference"));
            EditorGUILayout.PropertyField(_languageId, new GUIContent("Language ID"));
            EditorGUILayout.PropertyField(_exaggeration);
            EditorGUILayout.PropertyField(_maxNewTokens, new GUIContent("Max New Tokens"));
            EditorGUILayout.PropertyField(_repetitionPenalty, new GUIContent("Repetition Penalty"));

            EditorGUILayout.Space(5);

            // Output
            EditorGUILayout.LabelField("Output", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(_audioSource, new GUIContent("Audio Source"));
            EditorGUILayout.PropertyField(_autoPlay, new GUIContent("Auto Play"));

            EditorGUILayout.Space(10);

            // Preview / Generate section
            DrawPreviewSection(tts);

            EditorGUILayout.Space(5);

            // Events (foldout)
            _showEvents = EditorGUILayout.Foldout(_showEvents, "Events", true);
            if (_showEvents)
            {
                EditorGUILayout.PropertyField(_onSynthesis);
                EditorGUILayout.PropertyField(_onAudioClipReady);
                EditorGUILayout.PropertyField(_onError);
            }

            serializedObject.ApplyModifiedProperties();
        }

        private void DrawPreviewSection(TextToSpeech tts)
        {
            EditorGUILayout.LabelField("Preview", EditorStyles.boldLabel);

            EditorGUILayout.LabelField("Text");
            _previewText = EditorGUILayout.TextArea(_previewText, GUILayout.MinHeight(60));

            EditorGUILayout.Space(5);

            var canGenerate = Application.isPlaying && tts.IsReady && !_isGenerating;

            using (new EditorGUI.DisabledScope(!canGenerate))
            {
                if (GUILayout.Button(
                        _isGenerating ? "Generating..." : "Generate & Play",
                        GUILayout.Height(30)))
                {
                    GenerateAsync(tts);
                }
            }

            if (!Application.isPlaying)
            {
                EditorGUILayout.HelpBox("Enter Play Mode to generate audio.", MessageType.Info);
            }
            else if (_engine.objectReferenceValue == null)
            {
                EditorGUILayout.HelpBox("Assign a TTS Engine to generate audio.", MessageType.Warning);
            }
            else if (!tts.IsReady)
            {
                EditorGUILayout.HelpBox("Engine is loading...", MessageType.Info);
            }
        }

        private async void GenerateAsync(TextToSpeech tts)
        {
            if (string.IsNullOrWhiteSpace(_previewText))
                return;

            _isGenerating = true;
            Repaint();

            try
            {
                await tts.SynthesizeAsync(_previewText);
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[TextToSpeech Editor] {ex.Message}");
            }
            finally
            {
                _isGenerating = false;
                Repaint();
            }
        }
    }
}
