using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Editor
{
    /// <summary>
    /// Disposes all live InferenceEngine instances before domain reload to prevent
    /// native ONNX Runtime crashes during GC finalization. Also disposes on exit
    /// play mode as an additional safety net.
    /// </summary>
    [InitializeOnLoad]
    static class InferenceEngineDomainReloadHandler
    {
        static InferenceEngineDomainReloadHandler()
        {
            AssemblyReloadEvents.beforeAssemblyReload += OnBeforeAssemblyReload;
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
        }

        private static void OnBeforeAssemblyReload()
        {
            DisposeAllEngines("domain reload");
        }

        private static void OnPlayModeStateChanged(PlayModeStateChange state)
        {
            if (state == PlayModeStateChange.ExitingPlayMode)
                DisposeAllEngines("exiting play mode");
        }

        private static void DisposeAllEngines(string reason)
        {
            var count = InferenceEngineRuntimeBase.LiveRuntimes.Count;
            if (count > 0)
            {
                Debug.Log($"[InferenceEngine] Disposing {count} live engine(s) before {reason}");
                InferenceEngineRuntimeBase.DisposeAll();
            }
        }
    }
}
