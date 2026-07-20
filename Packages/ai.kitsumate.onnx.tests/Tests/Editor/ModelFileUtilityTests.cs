using System.Collections.Generic;
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Editor.Download;
using NUnit.Framework;

namespace KitsuMate.Onnx.Tests
{
    public class ModelFileUtilityTests
    {
        [TestCase("model.onnx")]
        [TestCase("model.ort")]
        [TestCase("nested/model.onnx")]
        [TestCase("nested/model.ort")]
        public void IsSupportedModelPath_AcceptsSupportedModelExtensions(string path)
        {
            Assert.IsTrue(ModelFileUtility.IsSupportedModelPath(path));
        }

        [Test]
        public void GetModelFilePatterns_ReturnsOnnxAndOrtPatterns()
        {
            var patterns = ModelFileUtility.GetModelFilePatterns("encoder_model", "_q4f16");

            CollectionAssert.AreEqual(
                new[] { "encoder_model_q4f16.onnx", "encoder_model_q4f16.ort" },
                patterns);
        }

        [Test]
        public void ResolveFiles_ResolvesOrtModelWithoutAddingExternalData()
        {
            var files = new List<RepoFile>
            {
                new("onnx/model_q4f16.ort", "model_q4f16.ort", 123),
            };

            var resolved = RepoFileClient.ResolveFiles(
                files,
                new[] { new FileDefinition("model", ModelFileUtility.GetModelFilePatterns("model", "_q4f16")) });

            Assert.That(resolved.TryGetValue("model", out var modelPath), Is.True);
            Assert.That(modelPath, Is.EqualTo("onnx/model_q4f16.ort"));
            Assert.That(resolved.ContainsKey("model_data"), Is.False);
        }

        [Test]
        public void ResolveFiles_AddsExternalDataForOnnxModel()
        {
            var files = new List<RepoFile>
            {
                new("onnx/model_q4f16.onnx", "model_q4f16.onnx", 123),
                new("onnx/model_q4f16.onnx_data", "model_q4f16.onnx_data", 456),
            };

            var resolved = RepoFileClient.ResolveFiles(
                files,
                new[] { new FileDefinition("model", ModelFileUtility.GetModelFilePatterns("model", "_q4f16")) });

            Assert.That(resolved.TryGetValue("model", out var modelPath), Is.True);
            Assert.That(modelPath, Is.EqualTo("onnx/model_q4f16.onnx"));
            Assert.That(resolved.TryGetValue("model_data", out var dataPath), Is.True);
            Assert.That(dataPath, Is.EqualTo("onnx/model_q4f16.onnx_data"));
        }
    }
}