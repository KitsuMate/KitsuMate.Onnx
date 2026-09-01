using System;
using System.IO;
using System.Linq;
using KitsuMate.Onnx.Motion.Kimodo;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Tests
{
    public class KimodoConstraintCompilerTests
    {
        [Test]
        public void Capabilities_DescribeConstrainedSomaModel()
        {
            KimodoModelCapabilities capabilities = KimodoConstraintCompiler.SomaRpV11Capabilities;
            Assert.AreEqual(60, capabilities.FrameCount);
            Assert.AreEqual(369, capabilities.MotionFeatureCount);
            Assert.AreEqual(30, capabilities.InternalJointCount);
            Assert.IsTrue(capabilities.SupportsSeparatedGuidance);
            Assert.IsTrue((capabilities.Constraints & KimodoConstraintCapabilities.FullBodyPose) != 0);
            Assert.IsTrue((capabilities.Constraints & KimodoConstraintCapabilities.RightFoot) != 0);
        }

        [Test]
        public void RawConditioning_ValidatesCopiesAndDetectsMask()
        {
            var observed = new float[60 * 369];
            var mask = new bool[60 * 369];
            observed[42] = 1.25f;
            mask[42] = true;
            var conditioning = new KimodoConditioning(observed, mask);
            observed[42] = 9f;
            mask[42] = false;
            Assert.AreEqual(1.25f, conditioning.ObservedMotion.Span[42]);
            Assert.IsTrue(conditioning.MotionMask.Span[42]);
            Assert.IsTrue(conditioning.HasConstraints);
            Assert.Throws<ArgumentException>(() => new KimodoConditioning(new float[1], new bool[1]));
        }

        [Test]
        public void SeparatedCfgBatch_UsesTextConstraintAndUnconditionalBranches()
        {
            var embeddingValues = Enumerable.Range(0, 4096).Select(index => index * 0.001f).ToArray();
            var embedding = new KimodoTextEmbedding(embeddingValues, copy: false);
            int motionLength = 60 * 369;
            var observed = new float[motionLength];
            var mask = new bool[motionLength];
            observed[7] = 3.5f;
            mask[7] = true;
            var conditioning = new KimodoConditioning(observed, mask, copy: false);
            var textBatch = new float[3 * 50 * 4096];
            var maskBatch = new bool[3 * motionLength];
            var observedBatch = new float[3 * motionLength];

            KimodoCfgBatchBuilder.Build(embedding, conditioning, textBatch, maskBatch, observedBatch);

            CollectionAssert.AreEqual(embeddingValues, textBatch.Take(4096).ToArray());
            Assert.IsTrue(textBatch.Skip(4096).All(value => value == 0f));
            Assert.IsFalse(maskBatch[7]);
            Assert.IsTrue(maskBatch[motionLength + 7]);
            Assert.IsFalse(maskBatch[2 * motionLength + 7]);
            Assert.AreEqual(3.5f, observedBatch[7]);
            Assert.AreEqual(3.5f, observedBatch[motionLength + 7]);
            Assert.AreEqual(3.5f, observedBatch[2 * motionLength + 7]);
        }

        [Test]
        [Category("Integration")]
        public void Compiler_MatchesPythonConditioningFixtures()
        {
            string root = FixtureRoot();
            Assert.That(Directory.Exists(root), Is.True, $"Kimodo conditioning fixtures are missing at {root}.");
            JObject manifest = JObject.Parse(File.ReadAllText(Path.Combine(root, "manifest.json")));
            var compiler = new KimodoConstraintCompiler();
            foreach (JObject record in manifest["cases"]!.Children<JObject>())
            {
                IKimodoConstraint constraint = ParseConstraint((JObject)record["semantic"]!);
                KimodoConditioning actual = compiler.Compile(new KimodoConstraintSet(constraint));
                float[] expected = ReadFloats(Path.Combine(root, record.Value<string>("observed")!));
                bool[] expectedMask = File.ReadAllBytes(Path.Combine(root, record.Value<string>("mask")!))
                    .Select(value => value != 0).ToArray();

                CollectionAssert.AreEqual(expectedMask, actual.MotionMask.ToArray(), $"Mask mismatch for {record.Value<string>("name")}");
                float maxError = 0f;
                double absoluteError = 0.0;
                ReadOnlySpan<float> values = actual.ObservedMotion.Span;
                for (int i = 0; i < expected.Length; i++)
                {
                    float error = Math.Abs(expected[i] - values[i]);
                    absoluteError += error;
                    if (error > maxError) maxError = error;
                }
                double mae = absoluteError / expected.Length;
                Assert.LessOrEqual(mae, 2e-6, $"{record.Value<string>("name")} MAE={mae}, max={maxError}");
                Assert.LessOrEqual(maxError, 2e-4f, $"{record.Value<string>("name")} max={maxError}, MAE={mae}");
            }
        }

        [Test]
        public void SemanticCompiler_RejectsInvalidFramesHeadingsAndConflicts()
        {
            var compiler = new KimodoConstraintCompiler();
            var outsideTimeline = new KimodoRootConstraint(new[] { 60 }, new[] { Vector2.zero });
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                compiler.Compile(new KimodoConstraintSet(outsideTimeline)));

            var zeroHeading = new KimodoRootConstraint(new[] { 0 }, new[] { Vector2.zero }, new[] { Vector2.zero });
            Assert.Throws<ArgumentException>(() =>
                compiler.Compile(new KimodoConstraintSet(zeroHeading)));

            var first = new KimodoRootConstraint(new[] { 12 }, new[] { new Vector2(1f, 2f) });
            var conflicting = new KimodoRootConstraint(new[] { 12 }, new[] { new Vector2(3f, 2f) });
            Assert.Throws<InvalidOperationException>(() =>
                compiler.Compile(new KimodoConstraintSet(first, conflicting)));
        }

        [Test]
        public void SemanticCompiler_ExplicitRootOverridesPoseDerivedContextAtSameFrame()
        {
            var positions = new Vector3[30];
            var rotations = Enumerable.Repeat(Quaternion.identity, 30).ToArray();
            positions[(int)KimodoJoint.Hips] = new Vector3(0f, 0.9f, 0f);
            // Deliberately derive a different heading from the pose's hip line.
            positions[(int)KimodoJoint.LeftLeg] = new Vector3(-0.1f, 0.8f, -0.2f);
            positions[(int)KimodoJoint.RightLeg] = new Vector3(0.1f, 0.8f, 0.2f);
            var root = new KimodoRootConstraint(
                new[] { 0 }, new[] { new Vector2(4f, 7f) }, new[] { new Vector2(1f, 0f) });
            var effector = new KimodoEndEffectorConstraint(
                new[] { 0 }, KimodoEndEffectors.LeftHand, positions, rotations,
                new[] { new Vector2(-3f, 2f) });

            KimodoConditioning result = new KimodoConstraintCompiler().Compile(
                new KimodoConstraintSet(effector, root));

            Assert.IsTrue(result.MotionMask.Span[0]);
            Assert.IsTrue(result.MotionMask.Span[2]);
            Assert.IsTrue(result.MotionMask.Span[3]);
            Assert.IsTrue(result.MotionMask.Span[4]);
            Assert.AreEqual((4f - KimodoSomaRuntimeData.Mean[0]) / KimodoSomaRuntimeData.Scale[0],
                result.ObservedMotion.Span[0], 1e-6f);
            Assert.AreEqual((7f - KimodoSomaRuntimeData.Mean[2]) / KimodoSomaRuntimeData.Scale[2],
                result.ObservedMotion.Span[2], 1e-6f);
            Assert.AreEqual((1f - KimodoSomaRuntimeData.Mean[3]) / KimodoSomaRuntimeData.Scale[3],
                result.ObservedMotion.Span[3], 1e-6f);
            Assert.AreEqual((0f - KimodoSomaRuntimeData.Mean[4]) / KimodoSomaRuntimeData.Scale[4],
                result.ObservedMotion.Span[4], 1e-6f);
        }

        private static IKimodoConstraint ParseConstraint(JObject semantic)
        {
            int[] frames = semantic["frame_indices"]!.Values<int>().ToArray();
            switch (semantic.Value<string>("type"))
            {
                case "root":
                    return new KimodoRootConstraint(frames, ReadVector2Array((JArray)semantic["positions_xz"]!), ReadVector2Array((JArray)semantic["headings"]!));
                case "fullbody":
                    return new KimodoFullBodyConstraint(frames, ReadVector3Tensor((JArray)semantic["global_positions"]!), ReadVector2Array((JArray)semantic["smooth_root_positions_xz"]!));
                case "end_effector":
                    KimodoEndEffectors effectors = KimodoEndEffectors.None;
                    foreach (string name in semantic["effectors"]!.Values<string>())
                        effectors |= (KimodoEndEffectors)Enum.Parse(typeof(KimodoEndEffectors), name);
                    return new KimodoEndEffectorConstraint(
                        frames,
                        effectors,
                        ReadVector3Tensor((JArray)semantic["global_positions"]!),
                        ReadQuaternionTensor((JArray)semantic["global_rotations_xyzw"]!),
                        ReadVector2Array((JArray)semantic["smooth_root_positions_xz"]!));
                default:
                    throw new InvalidDataException("Unknown fixture constraint type.");
            }
        }

        private static Vector2[] ReadVector2Array(JArray array) => array.Children<JArray>()
            .Select(value => new Vector2(value[0]!.Value<float>(), value[1]!.Value<float>())).ToArray();

        private static Vector3[] ReadVector3Tensor(JArray frames) => frames.Children<JArray>()
            .SelectMany(frame => frame.Children<JArray>())
            .Select(value => new Vector3(value[0]!.Value<float>(), value[1]!.Value<float>(), value[2]!.Value<float>())).ToArray();

        private static Quaternion[] ReadQuaternionTensor(JArray frames) => frames.Children<JArray>()
            .SelectMany(frame => frame.Children<JArray>())
            .Select(value => new Quaternion(value[0]!.Value<float>(), value[1]!.Value<float>(), value[2]!.Value<float>(), value[3]!.Value<float>())).ToArray();

        private static float[] ReadFloats(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            var result = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, result, 0, bytes.Length);
            return result;
        }

        private static string FixtureRoot() => Path.GetFullPath(Path.Combine(
            Application.dataPath, "..", "Packages", "ai.kitsumate.onnx.motion.tests", "Tests", "Fixtures",
            "KimodoConstraints"));
    }
}
