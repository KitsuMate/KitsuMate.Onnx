using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Asr.Sentis.Tests
{
    public sealed class SentisWhisperTests
    {
        [Test]
        public void EmptyModelSet_ReportsEveryRequiredAsset()
        {
            var modelSet = ScriptableObject.CreateInstance<SentisWhisperModelSet>();
            try
            {
                ModelValidationResult validation = modelSet.Validate(new ModelValidationContext(null));
                Assert.That(validation.IsValid, Is.False);
                Assert.That(validation.Diagnostics, Has.Some.Matches<ModelDiagnostic>(item => item.Code == "missing_mel"));
                Assert.That(validation.Diagnostics, Has.Some.Matches<ModelDiagnostic>(item => item.Code == "missing_encoder"));
                Assert.That(validation.Diagnostics, Has.Some.Matches<ModelDiagnostic>(item => item.Code == "missing_decoder"));
                Assert.That(validation.Diagnostics, Has.None.Matches<ModelDiagnostic>(
                    item => item.Code == "missing_cached_decoder"));
                Assert.That(validation.Diagnostics, Has.Some.Matches<ModelDiagnostic>(item => item.Code == "missing_tokenizer"));
            }
            finally
            {
                Object.DestroyImmediate(modelSet);
            }
        }
    }
}
