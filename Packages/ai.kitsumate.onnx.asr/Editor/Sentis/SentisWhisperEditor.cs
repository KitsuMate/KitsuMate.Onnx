using System.Collections.Generic;
using KitsuMate.Onnx.Download;
using KitsuMate.Onnx.Editor.Download;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;
namespace KitsuMate.Onnx.Asr.Sentis.Editor
{
    [CustomEditor(typeof(SentisWhisperModelSet))]
    public sealed class SentisWhisperModelSetEditor : KitsuMate.Onnx.Editor.ModelSetEditor
    {
        protected override bool UsesSentis => true;
        protected override void Installed(DownloadedModel installation) => ImportedModelGraphs.Apply(
            (ModelSet)target, installation, new Dictionary<string, string> { { "mel", "melProcessor" }, { "encoder", "encoder" }, { "decoder", "decoder" }, { "decoder-with-past", "decoderWithPast" } }, imported =>
            {
                var source = CreateInstance<UnityAiInferenceModelAsset>();
                source.SetModelAsset((ModelAsset)imported);
                return source;
            });
    }
    [CustomEditor(typeof(SentisWhisperEngine))]
    public sealed class SentisWhisperEngineEditor : KitsuMate.Onnx.Asr.Editor.AsrInferenceEditor
    {
        protected override void DrawEngineInspector() => DrawDefaultInspector();
    }
}
