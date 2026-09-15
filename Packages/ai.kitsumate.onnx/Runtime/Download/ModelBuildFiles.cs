#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;

namespace KitsuMate.Onnx.Download
{
    /// <summary>Tracks completed build copies using file metadata, without reading model contents.</summary>
    public static class ModelBuildFiles
    {
        public static void Move(string source, string destination)
        {
            source = Path.GetFullPath(source);
            destination = Path.GetFullPath(destination);
            var comparison = UnityEngine.Application.platform == UnityEngine.RuntimePlatform.WindowsEditor
                ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
            if (string.Equals(source, destination, comparison)) return;
            if (File.Exists(destination)) throw new IOException($"A file already exists at {destination}. Choose another destination or remove the duplicate first.");
            Directory.CreateDirectory(Path.GetDirectoryName(destination));
            string projectSource = FileUtil.GetProjectRelativePath(source.Replace('\\', '/'));
            string projectDestination = FileUtil.GetProjectRelativePath(destination.Replace('\\', '/'));
            if (projectSource.StartsWith("Assets/", StringComparison.Ordinal) &&
                projectDestination.StartsWith("Assets/", StringComparison.Ordinal) && File.Exists(source + ".meta"))
            {
                AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                string error = AssetDatabase.MoveAsset(projectSource, projectDestination);
                if (!string.IsNullOrEmpty(error)) throw new IOException(error);
            }
            else
            {
                bool hasMetadata = File.Exists(source + ".meta");
                if (hasMetadata && File.Exists(destination + ".meta"))
                    throw new IOException($"Asset metadata already exists at {destination}.meta.");
                File.Move(source, destination);
                try { if (hasMetadata) File.Move(source + ".meta", destination + ".meta"); }
                catch { File.Move(destination, source); throw; }
            }
        }

        public static string DirectoryFor(ModelSet set)
        {
            string path = AssetDatabase.GetAssetPath(set);
            if (string.IsNullOrEmpty(path)) return string.Empty;
            return Path.GetDirectoryName(path).Replace('\\', '/') + "/" + Path.GetFileNameWithoutExtension(path) + " Files";
        }

        public static bool Matches(string source, string destination)
        {
            var original = new FileInfo(source);
            var copy = new FileInfo(destination);
            return original.Exists && copy.Exists && original.Length == copy.Length &&
                original.LastWriteTimeUtc == copy.LastWriteTimeUtc;
        }

        public static bool Copy(string source, string destination)
        {
            if (Matches(source, destination)) return false;
            Directory.CreateDirectory(Path.GetDirectoryName(destination));
            // A failed copy must not look complete on the next attempt.
            string temporary = destination + ".copying~";
            try
            {
                File.Copy(source, temporary, true);
                File.SetLastWriteTimeUtc(temporary, File.GetLastWriteTimeUtc(source));
                if (File.Exists(destination)) File.Replace(temporary, destination, null);
                else File.Move(temporary, destination);
            }
            finally { if (File.Exists(temporary)) File.Delete(temporary); }
            return true;
        }
    }
}
#endif
