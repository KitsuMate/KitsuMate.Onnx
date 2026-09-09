using UnityEngine;

namespace KitsuMate.Onnx.LipSync
{
    /// <summary>Applies detected speech shapes without owning inference or audio playback.</summary>
    public abstract class VisemeTarget : MonoBehaviour
    {
        public abstract void ApplyViseme(VisemeFrame frame);
        public abstract void ResetVisemes();
    }
}
