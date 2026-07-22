using KitsuMate.Onnx.Editor;
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
    }
}
