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

        [Test]
        public void NonJsonTokenizer_RequiresVocabulary()
        {
            var modelSet = ScriptableObject.CreateInstance<UnityAiInferenceTextEmbeddingModelSet>();
            var tokenizer = new TextAsset("version: 0.2\nmerges: []");
            try
            {
                typeof(UnityAiInferenceTextEmbeddingModelSet)
                    .GetField("tokenizerModel", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
                    .SetValue(modelSet, tokenizer);

                ModelValidationResult validation = modelSet.Validate(new ModelValidationContext(null));
                Assert.That(validation.Diagnostics, Has.Some.Matches<ModelDiagnostic>(x => x.Code == "missing_vocabulary"));
            }
            finally
            {
                Object.DestroyImmediate(tokenizer);
                Object.DestroyImmediate(modelSet);
            }
        }
    }
}
