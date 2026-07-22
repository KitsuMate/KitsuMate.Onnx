using System.Linq;
using System.Threading.Tasks;
using KitsuMate.Onnx.Editor.Download;
using NUnit.Framework;

namespace KitsuMate.Onnx.Tests
{
    public sealed class ModelDownloaderScanningTests
    {
        [Test]
        [Category("Integration")]
        public async Task ScansOnnxCommunityModelVariants()
        {
            var request = new ModelDownloadRequest(
                "onnx-community/all-MiniLM-L6-v2-ONNX",
                "aff7a1dc4e8a1ea593e6ea21e95c22ef0a25966f",
                string.Empty,
                "Assets/StreamingAssets/KitsuMateModels");

            var variants = await ModelDownloader.GetVariantsAsync(request);

            Assert.That(variants, Is.EquivalentTo(new[] { "default", "fp16", "q4", "q4f16" }));
        }

        [Test]
        [Category("Integration")]
        public async Task ScansOnnxCommunityWhisperVariants()
        {
            var request = new ModelDownloadRequest(
                "onnx-community/whisper-tiny",
                "ff4177021cc41f7db950912b73ea4fdf7d01d8e7",
                string.Empty,
                "Assets/StreamingAssets/KitsuMateModels",
                "whisper");

            var variants = await ModelDownloader.GetVariantsAsync(request);

            Assert.That(variants.Contains("fp32"), Is.True);
            Assert.That(variants.Contains("bnb4"), Is.True);
            Assert.That(variants.Contains("q4"), Is.True);
        }
    }
}
