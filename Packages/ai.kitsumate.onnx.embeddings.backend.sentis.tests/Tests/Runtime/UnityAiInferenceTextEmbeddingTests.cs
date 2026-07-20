using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings.UnityAiInference.Tests
{
    public sealed class UnityAiInferenceTextEmbeddingTests
    {
        [Test]
        public void EmptyModelSet_ReportsItsMissingEngineSpecificInputs()
        {
            var modelSet = ScriptableObject.CreateInstance<UnityAiInferenceTextEmbeddingModelSet>();
            try
            {
                ModelValidationResult validation = modelSet.Validate(new ModelValidationContext(null));
                Assert.That(validation.IsValid, Is.False);
                Assert.That(validation.Diagnostics, Has.Some.Matches<ModelDiagnostic>(x => x.Code == "missing_model"));
                Assert.That(validation.Diagnostics, Has.Some.Matches<ModelDiagnostic>(x => x.Code == "missing_tokenizer"));
            }
            finally
            {
                Object.DestroyImmediate(modelSet);
            }
        }
    }
}
