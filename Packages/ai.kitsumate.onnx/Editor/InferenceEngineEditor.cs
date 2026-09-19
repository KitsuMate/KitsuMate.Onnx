using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Editor
{
    public abstract class InferenceEngineEditor<TRequest, TResult> : UnityEditor.Editor
    {
        private InferencePreviewSession<TRequest, TResult> session;
        private string error;
        private bool active;
        private bool hasResult;
        protected TResult Result { get; private set; }
        protected abstract object TestInputs { get; }
        protected InferenceEngine<TRequest, TResult> Engine => (InferenceEngine<TRequest, TResult>)target;
        protected abstract void DrawTestInputs();
        // Called synchronously before starting work so implementations can snapshot their inputs.
        protected abstract Func<CancellationToken, Task<TRequest>> CaptureRequest();
        protected abstract void DrawTestResult();
        protected virtual void AcceptResult(TResult result) { }
        protected virtual void ClearResult() { }
        protected virtual void DrawEngineInspector() => DrawDefaultInspector();

        protected virtual void OnEnable()
        {
            active = true;
            session = new InferencePreviewSession<TRequest, TResult>();
            InferenceTestPreferences.Load(target, TestInputs);
            InferenceEngineDomainReloadHandler.PreviewClosing += Close;
            EditorApplication.playModeStateChanged += PlayModeChanged;
            EditorApplication.update += UpdatePreview;
        }

        protected virtual void OnDisable()
        {
            Close();
            InferenceEngineDomainReloadHandler.PreviewClosing -= Close;
            EditorApplication.playModeStateChanged -= PlayModeChanged;
            EditorApplication.update -= UpdatePreview;
        }

        private void UpdatePreview() { if (session.IsBusy) Repaint(); }
        private void PlayModeChanged(PlayModeStateChange state)
        {
            if (state == PlayModeStateChange.ExitingEditMode) Close();
            if (state == PlayModeStateChange.EnteredEditMode)
            {
                session = new InferencePreviewSession<TRequest, TResult>();
                active = true;
            }
        }

        private async void Close()
        {
            if (!active) return;
            active = false;
            ClearResult();
            Result = default;
            hasResult = false;
            try { await session.CloseAsync(); }
            catch (Exception exception) { Debug.LogException(exception); }
        }

        public sealed override void OnInspectorGUI()
        {
            using (new EditorGUI.DisabledScope(session.IsBusy)) DrawEngineInspector();
            EditorGUILayout.Space(12);
            EditorGUILayout.LabelField("Test Inference", EditorStyles.boldLabel);
            if (targets.Length != 1)
            {
                EditorGUILayout.HelpBox("Select one engine to test inference.", MessageType.Info);
                return;
            }
            if (EditorApplication.isPlayingOrWillChangePlaymode)
                EditorGUILayout.HelpBox("Exit Play Mode to use this inference test.", MessageType.Info);
            using (new EditorGUI.DisabledScope(session.IsBusy || EditorApplication.isPlayingOrWillChangePlaymode))
            {
                EditorGUI.BeginChangeCheck();
                DrawTestInputs();
                if (EditorGUI.EndChangeCheck()) InferenceTestPreferences.Save(target, TestInputs);
                if (GUILayout.Button("Run")) RunTest();
            }
            using (new EditorGUI.DisabledScope(!session.IsBusy))
                if (GUILayout.Button("Cancel")) session.Cancel();
            using (new EditorGUI.DisabledScope(session.IsBusy || !session.IsLoaded))
                if (GUILayout.Button("Unload")) Unload();
            EditorGUILayout.LabelField("Status", session.Status);
            EditorGUILayout.LabelField("Timing", $"Load {session.LoadMilliseconds:F0} ms / Run {session.RunMilliseconds:F0} ms");
            if (!string.IsNullOrEmpty(error)) EditorGUILayout.HelpBox(error, MessageType.Error);
            if (hasResult) DrawTestResult();
        }

        private async void RunTest()
        {
            try
            {
                error = null;
                var request = CaptureRequest();
                InferenceTestPreferences.Save(target, TestInputs);
                ClearResult();
                Result = default;
                hasResult = false;
                var current = session;
                var engine = Engine;
                TResult result = await current.RunAsync(Fingerprint(engine), engine.CreateRuntimeAsync, request);
                if (!active || current != session) return;
                Result = result;
                AcceptResult(result);
                hasResult = true;
            }
            catch (OperationCanceledException) { }
            catch (Exception exception) { if (active) error = exception.Message; Debug.LogException(exception); }
            finally { if (active) Repaint(); }
        }

        private async void Unload()
        {
            try { await session.UnloadAsync(); }
            catch (Exception exception) { error = exception.Message; }
        }

        public static string Fingerprint(InferenceEngineBase engine)
        {
            string Describe(UnityEngine.Object value)
            {
                if (value == null) return "";
                string path = AssetDatabase.GetAssetPath(value);
                return EditorJsonUtility.ToJson(value) + (string.IsNullOrEmpty(path) ? "" : AssetDatabase.GetAssetDependencyHash(path).ToString());
            }
            return Hash128.Compute(Describe(engine) + Describe(engine.ModelSet) + Describe(engine.Backend) +
                Describe(OnnxSettings.Load())).ToString();
        }
    }

    public static class InferenceTestPreferences
    {
        public static string Key(UnityEngine.Object engine) => "KitsuMate.InferenceTest." +
            Hash128.Compute(Application.dataPath) + "." + AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(engine));
        public static void Load(UnityEngine.Object engine, object inputs)
        {
            string json = EditorPrefs.GetString(Key(engine), "");
            if (string.IsNullOrEmpty(json)) return;
            try { JsonUtility.FromJsonOverwrite(json, inputs); }
            catch (ArgumentException) { EditorPrefs.DeleteKey(Key(engine)); }
        }
        public static void Save(UnityEngine.Object engine, object inputs) => EditorPrefs.SetString(Key(engine), JsonUtility.ToJson(inputs));
        public static T Asset<T>(string guid) where T : UnityEngine.Object =>
            string.IsNullOrEmpty(guid) ? null : AssetDatabase.LoadAssetAtPath<T>(AssetDatabase.GUIDToAssetPath(guid));
        public static string AssetField<T>(string label, string guid) where T : UnityEngine.Object
        {
            var selected = EditorGUILayout.ObjectField(label, Asset<T>(guid), typeof(T), false);
            return selected == null ? "" : AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(selected));
        }
        public static void SaveText(string text, string extension)
        {
            string path = EditorUtility.SaveFilePanel("Save inference result", "", "inference-result", extension);
            if (!string.IsNullOrEmpty(path)) File.WriteAllText(path, text);
        }
    }
}
