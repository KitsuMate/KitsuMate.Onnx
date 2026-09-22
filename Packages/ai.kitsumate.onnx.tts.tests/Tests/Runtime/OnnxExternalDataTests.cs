using System;
using System.IO;
using KitsuMate.Onnx;
using NUnit.Framework;

namespace KitsuMate.Onnx.Tts.Tests
{
    public sealed class OnnxExternalDataTests
    {
        [Test]
        public void NestedExternalDataPathAndRangeAreReadFromGraph()
        {
            using var fixture = ExternalDataFixture.Create();
            string graph = fixture.GraphPath;
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
            using var fixture = ExternalDataFixture.Create();
            string graph = fixture.GraphPath;
            Assert.Throws<InvalidDataException>(() => OnnxLightweightMetadataReader.ResolveExternalDataPath(
                graph, new OnnxLightweightMetadataReader.ExternalDataReference("../other.bin", 0, 1)));
            Assert.Throws<InvalidDataException>(() => OnnxLightweightMetadataReader.ResolveExternalDataPath(
                graph, new OnnxLightweightMetadataReader.ExternalDataReference(
                    OnnxLightweightMetadataReader.Read(graph).ExternalData[0].Location, long.MaxValue, 1)));
        }

    }

    internal sealed class ExternalDataFixture : IDisposable
    {
        private const string GraphBase64 =
            "CAgSFEtpdHN1TWF0ZS5Pbm54LlRlc3RzOqIBChwKBWlucHV0CgZ3ZWlnaHQSBm91dHB1dCIDQWRk" +
            "EhVleHRlcm5hbC1kYXRhLWZpeHR1cmUqQAgBEAFCBndlaWdodGoWCghsb2NhdGlvbhIKZmlyc3Qu" +
            "ZGF0YWoLCgZvZmZzZXQSATBqCwoGbGVuZ3RoEgE0cAFaEwoFaW5wdXQSCgoICAESBAoCCAFiFAoG" +
            "b3V0cHV0EgoKCAgBEgQKAggBQgQKABAN";

        private ExternalDataFixture(string directoryPath)
        {
            DirectoryPath = directoryPath;
            GraphPath = Path.Combine(directoryPath, "first.onnx");
        }

        public string DirectoryPath { get; }
        public string GraphPath { get; }

        public static ExternalDataFixture Create()
        {
            string directory = Path.Combine(Path.GetTempPath(), "kitsumate-onnx-external-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);
            var fixture = new ExternalDataFixture(directory);
            File.WriteAllBytes(fixture.GraphPath, Convert.FromBase64String(GraphBase64));
            File.WriteAllBytes(Path.Combine(directory, "first.data"), new byte[] { 0, 0, 128, 63 });
            return fixture;
        }

        public void Dispose()
        {
            if (Directory.Exists(DirectoryPath)) Directory.Delete(DirectoryPath, true);
        }
    }
}
