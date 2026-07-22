using UnityEngine;

namespace KitsuMate.Onnx.LipSync.Uni2005
{
    [CreateAssetMenu(fileName = "Uni2005Engine", menuName = "KitsuMate/ONNX/LipSync/Uni2005 Engine")]
    public sealed class Uni2005Engine : LipSyncEngine
    {
        [SerializeField] private Uni2005ModelSet modelSet;
        [SerializeField, Range(64, 2048)] private int framesPerBatch = 512;
        [SerializeField, Range(0f, 1f)] private float volumeThreshold = 0.01f;
        [SerializeField] private bool verboseLogging;
        public override ModelSet ModelSet => modelSet;
        protected override InferenceEngineRuntime<LipSyncRequest, VisemeTimeline> CreateRuntime() => new Uni2005EngineRuntime(modelSet, framesPerBatch, volumeThreshold, verboseLogging);
    }
}
