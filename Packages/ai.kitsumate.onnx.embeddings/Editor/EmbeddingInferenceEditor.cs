using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using KitsuMate.Onnx.Editor;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings.Editor
{
    public abstract class EmbeddingInferenceEditor : InferenceEngineEditor<EmbeddingRequest, EmbeddingResult>
    {
        [Serializable] private sealed class Inputs
        {
            public string Text = "A cat is sleeping.\nAn animal is resting.";
            public EmbeddingPurpose Purpose;
            public EmbeddingPooling Pooling;
            public int Normalization, MaxTokens;
        }
        [Serializable] public sealed class Vector { public float[] Values; }
        [Serializable] public sealed class Export { public Vector[] Vectors; }
        private readonly Inputs input = new();
        private Vector2 scroll;
        private int vectorIndex;
        private string output;
        protected override object TestInputs => input;
        protected override void DrawTestInputs()
        {
            EditorGUILayout.LabelField("Texts (one per line)");
            input.Text = EditorGUILayout.TextArea(input.Text, GUILayout.MinHeight(60));
            input.Purpose = (EmbeddingPurpose)EditorGUILayout.EnumPopup("Purpose", input.Purpose);
            input.Pooling = (EmbeddingPooling)EditorGUILayout.EnumPopup("Pooling", input.Pooling);
            input.Normalization = EditorGUILayout.Popup("Normalize", input.Normalization, new[] { "Model default", "Yes", "No" });
            input.MaxTokens = EditorGUILayout.IntField("Max tokens (0 = default)", input.MaxTokens);
        }
        protected override Func<CancellationToken, Task<EmbeddingRequest>> CaptureRequest()
        {
            string[] texts = input.Text.Split('\n').Select(text => text.Trim()).Where(text => text.Length > 0).ToArray();
            if (texts.Length == 0) throw new ArgumentException("Enter at least one text.");
            if (input.MaxTokens < 0) throw new ArgumentException("Max tokens cannot be negative.");
            var request = new EmbeddingRequest(texts) { Purpose = input.Purpose, Pooling = input.Pooling,
                Normalize = input.Normalization == 0 ? null : input.Normalization == 1,
                MaxTokenCount = input.MaxTokens == 0 ? null : input.MaxTokens };
            return _ => Task.FromResult(request);
        }
        public static string ExportJson(EmbeddingResult result) => JsonUtility.ToJson(new Export {
            Vectors = (result.Batch ?? new[] { result.Embedding }).Select(values => new Vector { Values = values }).ToArray() }, true);
        protected override void AcceptResult(EmbeddingResult result) { output = ExportJson(result); vectorIndex = 0; }
        protected override void DrawTestResult()
        {
            var vectors = Result.Batch ?? new[] { Result.Embedding };
            EditorGUILayout.LabelField("Result", $"{vectors.Length} vectors, {Result.Dimension} dimensions");
            vectorIndex = EditorGUILayout.IntSlider("Vector", vectorIndex, 0, vectors.Length - 1);
            scroll = EditorGUILayout.BeginScrollView(scroll, GUILayout.MaxHeight(240));
            EditorGUILayout.TextArea(string.Join(", ", vectors[vectorIndex].Select(v => v.ToString("G6", System.Globalization.CultureInfo.InvariantCulture))));
            for (int i = 0; i < vectors.Length; i++)
                for (int j = i + 1; j < vectors.Length; j++)
                    EditorGUILayout.LabelField($"Similarity {i + 1} / {j + 1}", new EmbeddingResult(vectors[i]).CosineSimilarity(vectors[j]).ToString("F4"));
            EditorGUILayout.EndScrollView();
            if (GUILayout.Button("Save JSON")) InferenceTestPreferences.SaveText(output, "json");
        }
    }
}
