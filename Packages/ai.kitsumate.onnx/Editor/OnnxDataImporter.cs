#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace KitsuMate.Onnx.Editor
{
    /// <summary>
    /// Importer for .onnx_data files (external weight data for ONNX models).
    /// Sentis does not handle .onnx_data, so this is always registered as primary.
    /// </summary>
    [ScriptedImporter(2, new[] { "onnx_data" }, importQueueOffset: 101)]
    public class OnnxDataImporter : ScriptedImporter
    {
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var fileInfo = new FileInfo(ctx.assetPath);
            string correspondingModel = GetCorrespondingModelPath(ctx.assetPath);
            string modelName = correspondingModel != null 
                ? Path.GetFileName(correspondingModel) 
                : null;
            
            string content;
            if (modelName != null)
            {
                content = $"ONNX External Data File\n" +
                    $"Size: {FormatFileSize(fileInfo.Length)}\n" +
                    $"Model: {modelName}\n\n" +
                    "This file contains weight data that is merged into the model at import time.";
            }
            else
            {
                content = $"ONNX External Data File\n" +
                    $"Size: {FormatFileSize(fileInfo.Length)}\n\n" +
                        "⚠ WARNING: No corresponding model file found!\n\n" +
                        "This file should be placed in the same directory as the corresponding .onnx or .ort file.\n" +
                        "Expected model name: " + Path.GetFileNameWithoutExtension(ctx.assetPath) + ".onnx or " + Path.GetFileNameWithoutExtension(ctx.assetPath) + ".ort";
            }
            
            var textAsset = new TextAsset(content);
            textAsset.name = Path.GetFileNameWithoutExtension(ctx.assetPath);
            
            var icon = LoadOnnxIcon();
            ctx.AddObjectToAsset("OnnxExternalData", textAsset, icon);
            ctx.SetMainObject(textAsset);
            
            if (correspondingModel == null)
            {
                ctx.LogImportWarning(
                    $"External data file '{ctx.assetPath}' has no corresponding .onnx or .ort file in the same directory.");
            }
        }
        
        private static string GetCorrespondingModelPath(string onnxDataPath)
        {
            string directory = Path.GetDirectoryName(onnxDataPath);
            string fileName = Path.GetFileNameWithoutExtension(onnxDataPath);
            
            // model.onnx_data -> model.onnx or model.ort (fileName = "model")
            // model_external.onnx_data -> model_external.onnx or model_external.ort
            foreach (var modelPath in ModelFileUtility.GetModelFilePatterns(fileName))
            {
                var candidatePath = Path.Combine(directory, modelPath);
                if (File.Exists(candidatePath))
                    return candidatePath;
            }
            
            // model.onnx.data / model.ort.data case
            if (ModelFileUtility.IsSupportedModelPath(fileName) && File.Exists(Path.Combine(directory, fileName)))
                return Path.Combine(directory, fileName);
            
            return null;
        }
        
        private static string FormatFileSize(long bytes)
        {
            string[] sizes = { "B", "KB", "MB", "GB" };
            int order = 0;
            double size = bytes;
            while (size >= 1024 && order < sizes.Length - 1)
            {
                order++;
                size /= 1024;
            }
            return $"{size:0.##} {sizes[order]}";
        }

        private static Texture2D LoadOnnxIcon()
        {
            const string iconGuid = "16c6d06ecc04b0444970869c7e0203e6";
            string iconPath = AssetDatabase.GUIDToAssetPath(iconGuid);
            if (string.IsNullOrEmpty(iconPath))
            {
                return null;
            }
            return AssetDatabase.LoadAssetAtPath<Texture2D>(iconPath);
        }
    }
}
#endif
