#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using KitsuMate.Onnx.Motion.Kimodo;
using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Tests
{
    public sealed class KimodoInferenceTests
    {
        [Test]
        [Category("Integration")]
        public Task Kimodo_GeneratesDifferentLengthsInOneCpuSession() =>
            Generate(OnnxExecutionProvider.Cpu, new[] { 2, 60, 300, 30 }, 2);

        [TestCase(300), TestCase(900), TestCase(1800), TestCase(5400)]
        [Explicit("Requires WebGPU and the downloaded Kimodo fixture.")]
        [Timeout(600000)]
        [Category("Integration")]
        public Task Kimodo_GeneratesLongSequencesOnWebGpu(int frames) =>
            Generate(OnnxExecutionProvider.WebGpu, new[] { frames }, 25);

        private static async Task Generate(OnnxExecutionProvider provider, int[] lengths, int steps)
        {
            string path = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "KitsuMateOnnxFixtures/kimodo/model_fp16.onnx"));
            if (!File.Exists(path)) Assert.Ignore("Download the Kimodo integration fixture first.");
            var modelSet = ScriptableObject.CreateInstance<KimodoModelSet>();
            var embeddingSet = ScriptableObject.CreateInstance<PrecomputedEmbeddingModelSet>();
            var engine = ScriptableObject.CreateInstance<KimodoEngine>();
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            InferenceEngineRuntime<CharacterMotionRequest, CharacterMotionResult> runtime = null;
            try
            {
                backend.SetProviderOrder(provider);
                modelSet.MotionModel.ConfigureFile(path, string.Empty, null, null);
                modelSet.SetRequiredEmbedding(embeddingSet);
                typeof(KimodoEngine).GetField("modelSet", BindingFlags.Instance | BindingFlags.NonPublic).SetValue(engine, modelSet);
                runtime = await engine.CreateRuntimeAsync(backend);
                byte[] embeddingBytes = File.ReadAllBytes("Packages/ai.kitsumate.onnx.motion.tests/Tests/PlayMode/Resources/Llm2VecReference.bytes");
                var embedding = new float[4096];
                Buffer.BlockCopy(embeddingBytes, 0, embedding, 0, embeddingBytes.Length);
                foreach (int frames in lengths)
                {
                    var watch = System.Diagnostics.Stopwatch.StartNew();
                    CharacterMotionResult result = await runtime.RunAsync(new CharacterMotionRequest
                    {
                        Segments = new[] { new CharacterMotionSegment(new KimodoTextEmbedding(embedding),
                            new KimodoGenerationRequest(seed: 1, denoisingSteps: steps, frameCount: frames)) }
                    });
                    Assert.That(result.Motion.FrameCount, Is.EqualTo(frames));
                    Assert.That(result.Motion.RootPositions.All(p => float.IsFinite(p.x) && float.IsFinite(p.y) && float.IsFinite(p.z)));
                    Assert.That(result.Motion.BoneRotationDeltas.All(q => float.IsFinite(q.x) && float.IsFinite(q.y) && float.IsFinite(q.z) && float.IsFinite(q.w)));
                    if (steps >= 25)
                    {
                        float maxJoinDistance = 0, maxJoinAngle = 0;
                        int completed = CharacterMotionPlanner.NextWindowFrames(frames, 0);
                        while (completed < frames)
                        {
                            maxJoinDistance = Mathf.Max(maxJoinDistance, Vector3.Distance(result.Motion.RootPositions[completed - 1], result.Motion.RootPositions[completed]));
                            maxJoinAngle = Mathf.Max(maxJoinAngle, Quaternion.Angle(result.Motion.RootRotations[completed - 1], result.Motion.RootRotations[completed]));
                            completed += CharacterMotionPlanner.NextWindowFrames(frames - completed, 5);
                        }
                        TestContext.WriteLine($"Maximum join step: {maxJoinDistance:F4} m, {maxJoinAngle:F2} degrees");
                        Assert.Less(maxJoinDistance, .5f, "Walking fixture should not jump at a window boundary.");
                        Assert.Less(maxJoinAngle, 60f, "Walking fixture should not turn abruptly at a window boundary.");
                    }
                    TestContext.WriteLine($"{provider}: {frames} frames, {steps} steps, {watch.Elapsed.TotalSeconds:F2} seconds");
                }
            }
            finally
            {
                runtime?.Dispose();
                UnityEngine.Object.DestroyImmediate(backend);
                UnityEngine.Object.DestroyImmediate(engine);
                UnityEngine.Object.DestroyImmediate(modelSet);
                UnityEngine.Object.DestroyImmediate(embeddingSet);
            }
        }

        // Motion accepts prebaked embeddings; this test does not load a text encoder.
        private sealed class PrecomputedEmbeddingModelSet : ModelSet
        {
            public override string DisplayName => "Precomputed test embeddings";
            public override ModelIdentity Identity => new ModelIdentity("test", "embedding", "1", "test");
            public override ModelCapabilities Capabilities => default;
            public override bool IsComplete => true;
            public override IOnnxModelSource[] GetAllModels() => Array.Empty<IOnnxModelSource>();
            public override ModelValidationResult Validate(ModelValidationContext context) => new ModelValidationResult();
        }
    }
}
#endif
