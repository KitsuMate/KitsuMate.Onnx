using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using KitsuMate.Onnx.Download;
using UnityEngine;
using UnityEngine.Networking;

namespace KitsuMate.Onnx
{
    /// <summary>Owns the model configuration copy and companion objects used by one runtime.</summary>
    public sealed class ResolvedModelSet : IDisposable
    {
        private readonly List<UnityEngine.Object> owned = new();
        private readonly SynchronizationContext context = SynchronizationContext.Current;
        public ModelSet Model { get; private set; }
        internal ResolvedModelSet(ModelSet model) => Model = model;
        internal void Clone() => Model = Own(UnityEngine.Object.Instantiate(Model));

#if UNITY_EDITOR
        internal string CompanionAssetDirectory { get; set; }

        private T ImportCompanion<T>(DownloadedModel installation, string role) where T : UnityEngine.Object
        {
            string path = ModelDownloadPaths.Child(CompanionAssetDirectory, installation.GetFile(role).Path);
            bool changed = ModelBuildFiles.Copy(installation.GetPath(role), path);
            string assetPath = UnityEditor.FileUtil.GetProjectRelativePath(path.Replace('\\', '/'));
            UnityEditor.AssetDatabase.ImportAsset(Path.GetDirectoryName(assetPath).Replace('\\', '/'),
                UnityEditor.ImportAssetOptions.ForceSynchronousImport);
            if (changed || UnityEditor.AssetDatabase.LoadAssetAtPath<T>(assetPath) == null)
                UnityEditor.AssetDatabase.ImportAsset(assetPath, UnityEditor.ImportAssetOptions.ForceSynchronousImport |
                    UnityEditor.ImportAssetOptions.ForceUpdate);
            return UnityEditor.AssetDatabase.LoadAssetAtPath<T>(assetPath)
                ?? throw new InvalidDataException($"Could not import '{role}' as {typeof(T).Name}.");
        }
#endif

        private T Own<T>(T value) where T : UnityEngine.Object
        {
            value.hideFlags = HideFlags.HideAndDontSave;
            owned.Add(value);
            return value;
        }

        public TextFileReference ReadText(DownloadedModel installation, string role, bool optional = false)
        {
            if (optional && !System.Linq.Enumerable.Any(installation.Files, file => file.Role == role)) return new TextFileReference();
            return TextFileReference.FromFile(installation.GetPath(role));
        }

        public async Task<AudioClip> ReadAudioAsync(DownloadedModel installation, string role, CancellationToken cancellationToken)
        {
#if UNITY_EDITOR
            if (CompanionAssetDirectory != null) return ImportCompanion<AudioClip>(installation, role);
#endif
            using var request = UnityWebRequestMultimedia.GetAudioClip(new Uri(installation.GetPath(role)).AbsoluteUri, AudioType.WAV);
            var operation = request.SendWebRequest();
            while (!operation.isDone)
            {
                if (cancellationToken.IsCancellationRequested) { request.Abort(); cancellationToken.ThrowIfCancellationRequested(); }
                await Task.Yield();
            }
            cancellationToken.ThrowIfCancellationRequested();
            if (request.result != UnityWebRequest.Result.Success) throw new IOException(request.error);
            return Own(DownloadHandlerAudioClip.GetContent(request));
        }

        public void Dispose()
        {
            UnityEngine.Object[] objects = owned.ToArray();
            owned.Clear();
            void Destroy()
            {
                foreach (var value in objects)
                    if (value != null)
                    {
                        if (Application.isPlaying) UnityEngine.Object.Destroy(value);
                        else UnityEngine.Object.DestroyImmediate(value);
                    }
            }
            if (SynchronizationContext.Current == context) Destroy();
            else context.Post(_ => Destroy(), null);
        }
    }
}
