using System.IO;
using KitsuMate.Onnx;
using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Tests
{
    public sealed class OnnxExternalDataTests
    {
        [Test]
        public void NestedExternalDataPathAndRangeAreReadFromGraph()
        {
            string graph = FixtureGraph();
            var metadata = OnnxLightweightMetadataReader.Read(graph);
            Assert.That(metadata.ExternalData, Is.Not.Empty);
            var reference = metadata.ExternalData[0];
            Assert.That(reference.Location, Is.Not.Empty);
            string path = OnnxLightweightMetadataReader.ResolveExternalDataPath(graph, reference);
            Assert.That(new FileInfo(path).Length, Is.GreaterThanOrEqualTo(reference.Offset + reference.Length));
        }

        [Test]
        public void ExternalDataOutsideGraphDirectoryOrFileRangeIsRejected()
        {
            string graph = FixtureGraph();
            Assert.Throws<InvalidDataException>(() => OnnxLightweightMetadataReader.ResolveExternalDataPath(
                graph, new OnnxLightweightMetadataReader.ExternalDataReference("../other.bin", 0, 1)));
            Assert.Throws<InvalidDataException>(() => OnnxLightweightMetadataReader.ResolveExternalDataPath(
                graph, new OnnxLightweightMetadataReader.ExternalDataReference(
                    OnnxLightweightMetadataReader.Read(graph).ExternalData[0].Location, long.MaxValue, 1)));
        }

        private static string FixtureGraph() => Path.GetFullPath(Path.Combine(Application.dataPath, "..",
            "KitsuMateOnnxFixtures", "text-embedding", "all-minilm", "model_q4f16.onnx"));
    }
}
