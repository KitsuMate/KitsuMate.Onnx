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
            string path = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "KitsuMateOnnxFixtures",
                "text-embedding", "all-minilm", "model_q4f16.onnx"));
            Assert.That(File.Exists(path), Is.True, "Missing metadata fixture. Download the required CPU CI fixture set.");

            OnnxLightweightMetadataReader.Result result = OnnxLightweightMetadataReader.Read(path);
            Assert.That(result.Inputs.Select(x => x.Name), Contains.Item("input_ids"));
            Assert.That(result.Outputs, Is.Not.Empty);
        }
    }
}
#endif
