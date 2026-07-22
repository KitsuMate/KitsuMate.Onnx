using System.Linq;
using KitsuMate.Onnx.Asr.Editor;
using NUnit.Framework;

namespace KitsuMate.Onnx.Asr.Tests
{
    public sealed class WhisperModelDownloaderTests
    {
        [Test]
        public void Sources_UsePinnedRepositoriesForTinyAndBase()
        {
            Assert.That(WhisperModelDownloader.Sources.Select(source => source.Name),
                Is.EquivalentTo(new[] { "Whisper Tiny", "Whisper Base" }));
            Assert.That(WhisperModelDownloader.Sources.All(source => source.Repository.StartsWith("KitsuMate/")), Is.True);
            Assert.That(WhisperModelDownloader.Sources.All(source => !string.IsNullOrWhiteSpace(source.Revision)), Is.True);
        }
    }
}
