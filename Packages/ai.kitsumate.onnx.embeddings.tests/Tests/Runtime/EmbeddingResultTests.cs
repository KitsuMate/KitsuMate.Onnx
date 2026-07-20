using NUnit.Framework;

namespace KitsuMate.Onnx.Embeddings.Tests
{
    public class EmbeddingResultTests
    {
        [Test]
        public void CosineSimilarity_ReturnsExpectedValues()
        {
            var embedding = new EmbeddingResult(new[] { 1f, 0f, 0f });

            Assert.AreEqual(1f, embedding.CosineSimilarity(new[] { 1f, 0f, 0f }), 1e-5f);
            Assert.AreEqual(0f, embedding.CosineSimilarity(new[] { 0f, 1f, 0f }), 1e-5f);
            Assert.AreEqual(-1f, embedding.CosineSimilarity(new[] { -1f, 0f, 0f }), 1e-5f);
        }
    }
}
