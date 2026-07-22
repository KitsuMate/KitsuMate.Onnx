using NUnit.Framework;

namespace KitsuMate.Onnx.Tests
{
    public class CoreContractTests
    {
        [Test]
        public void DefaultSessionOptionsUsePortableOptimizationIntent()
        {
            Assert.AreEqual(OnnxOptimizationLevel.All, OnnxSessionOptions.Default.OptimizationLevel);
        }
    }
}
