using System.IO;
using KitsuMate.Onnx;
using NUnit.Framework;
using UnityEditor.PackageManager;

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

        private static string FixtureGraph()
        {
            string packagePath = PackageInfo.FindForAssembly(typeof(OnnxExternalDataTests).Assembly)?.resolvedPath;
            Assert.That(packagePath, Is.Not.Null, "Could not locate the TTS test package.");
            string graph = Path.Combine(packagePath, "Tests", "Fixtures", "ExternalData", "first.onnx");
            Assert.That(File.Exists(graph), Is.True, $"Required external-data fixture is missing: {graph}");
            return graph;
        }
    }
}
