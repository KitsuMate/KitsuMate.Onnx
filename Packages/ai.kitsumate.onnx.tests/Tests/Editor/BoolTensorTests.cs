using NUnit.Framework;

namespace KitsuMate.Onnx.Tests
{
    public class BoolTensorTests
    {
        [Test]
        public void BoolTensor_RoundTripsManagedData()
        {
            var values = new[] { true, false, true, true };
            using var tensor = OnnxTensor.FromArray(values, new[] { 2, 2 }, "mask");
            Assert.AreEqual(OnnxTensorElementType.Bool, tensor.ElementType);
            Assert.AreSame(values, tensor.AsBoolArray());
            CollectionAssert.AreEqual(new[] { 2, 2 }, tensor.Shape);
        }
    }
}
