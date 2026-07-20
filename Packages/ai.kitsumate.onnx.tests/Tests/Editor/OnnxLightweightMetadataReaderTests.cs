#if UNITY_EDITOR
using System.IO;
using System.Linq;
using KitsuMate.Onnx.Editor;
using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Tests
{
    public sealed class OnnxLightweightMetadataReaderTests
    {
        [Test]
        [Category("Integration")]
        public void Read_ExternalDataModel_ReturnsSchemaWithoutExternalWeights()
        {
            string path = Path.Combine(Application.dataPath, "KitsuMateOnnxFixtures", "metadata", "yolo10n_external.onnx");
            Assert.That(File.Exists(path), Is.True, "Missing metadata fixture. Hydrate the selected fixture profile.");

            OnnxLightweightMetadataReader.Result result = OnnxLightweightMetadataReader.Read(path);
            Assert.That(result.Inputs, Has.Count.EqualTo(1));
            Assert.That(result.Inputs.Select(x => x.Name), Contains.Item("images"));
            Assert.That(result.Outputs, Has.Count.EqualTo(1));
            Assert.That(result.Outputs.Select(x => x.Name), Contains.Item("output0"));
        }
    }
}
#endif
