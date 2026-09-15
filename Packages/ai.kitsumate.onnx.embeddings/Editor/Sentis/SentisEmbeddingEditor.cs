using System.Collections.Generic;
using KitsuMate.Onnx.Download;
using KitsuMate.Onnx.Editor.Download;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;
namespace KitsuMate.Onnx.Embeddings.Sentis.Editor
{
    [CustomEditor(typeof(SentisEmbeddingModelSet))]
    public sealed class SentisEmbeddingModelSetEditor : KitsuMate.Onnx.Editor.ModelSetEditor
    {
        protected override bool UsesSentis => true;
        protected override void Installed(DownloadedModel installation) => ImportedModelGraphs.Apply(
            (ModelSet)target, installation, new Dictionary<string, string> { { "model", "embeddingModel" } }, imported =>
            {
                var source = CreateInstance<UnityAiInferenceModelAsset>();
                source.SetModelAsset((ModelAsset)imported);
                return source;
            });
    }
    [CustomEditor(typeof(SentisEmbeddingEngine))]
    public sealed class SentisEmbeddingEngineEditor : KitsuMate.Onnx.Embeddings.Editor.EmbeddingInferenceEditor
    {
        protected override void DrawEngineInspector() => DrawDefaultInspector();
    }
}
