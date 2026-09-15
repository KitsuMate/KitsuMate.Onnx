using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using KitsuMate.Onnx.Motion.Kimodo;
using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Tests
{
    public class KimodoSequenceTests
    {
        [TestCase(2)] [TestCase(30)] [TestCase(59)] [TestCase(60)] [TestCase(61)]
        [TestCase(150)] [TestCase(299)] [TestCase(300)] [TestCase(301)]
        [TestCase(900)] [TestCase(1800)] [TestCase(5400)]
        public void Sequence_PreservesDurationAndBoundsEveryWindow(int total)
        {
            int calls = 0, reported = 0, generatedFrames = 0;
            var request = Request(total);
            request.Progress = new ImmediateProgress(value => reported = value);
            KimodoHumanoidMotion result = KimodoMotionSequence.Generate(request, (embedding, conditioning, generation, token) =>
            {
                Assert.That(generation.FrameCount, Is.InRange(2, 300));
                Assert.AreEqual(generation.FrameCount, conditioning.FrameCount);
                generatedFrames += generation.FrameCount;
                if (calls++ > 0)
                {
                    Assert.IsTrue(conditioning.MotionMask.Span[0]);
                    Assert.IsTrue(conditioning.MotionMask.Span[95 + 13 * 6]);
                }
                return Generate(embedding, conditioning, generation, token);
            }, default);
            Assert.AreEqual(total, result.FrameCount);
            Assert.AreEqual(total, reported);
            Assert.AreEqual(total + (calls - 1) * 5, generatedFrames);
            Assert.That(result.SourceMotion.All(float.IsFinite));
        }

        [Test]
        public void Sequence_MapsLateConstraintsAndChangesEmbedding()
        {
            var request = Request(305);
            var second = new KimodoTextEmbedding(new float[4096]);
            request.Segments = new[] { request.Segments[0], new CharacterMotionSegment(second, new KimodoGenerationRequest(frameCount: 40)) };
            request.Constraints = new KimodoConstraintSet(new KimodoRootConstraint(new[] { 344 }, new[] { new Vector2(4f, 2f) }));
            bool sawSecond = false, sawEnd = false;
            var result = KimodoMotionSequence.Generate(request, (embedding, conditioning, generation, token) =>
            {
                if (ReferenceEquals(embedding, second))
                {
                    sawSecond = true;
                    sawEnd = conditioning.MotionMask.Span[(generation.FrameCount - 1) * 369];
                }
                return Generate(embedding, conditioning, generation, token);
            }, default);
            Assert.IsTrue(sawSecond && sawEnd);
            Assert.Less(Vector3.Distance(result.SmoothedRootPositions[344], new Vector3(-4f, 0f, 2f)), 1e-5f);
        }

        [Test]
        public void Sequence_ShortPreviousMotionDoesNotRemoveOutputFrames()
        {
            var request = Request(3);
            float[] previous = Generate(null, KimodoConditioning.CreateEmpty(2), new KimodoGenerationRequest(frameCount: 2), default);
            request.PreviousMotion = KimodoMotionDecoder.Decode(previous, 2, null);
            request.PreviousMotion.SetSourceMotion(previous);
            request.Segments = new[] { request.Segments[0], new CharacterMotionSegment(new KimodoTextEmbedding(new float[4096]), new KimodoGenerationRequest(frameCount: 2)) };
            Assert.AreEqual(5, KimodoMotionSequence.Generate(request, Generate, default).FrameCount);
        }

        [Test]
        public void Sequence_CancellationStopsBeforeNextInference()
        {
            using var cancellation = new CancellationTokenSource();
            var request = Request(900);
            request.Progress = new ImmediateProgress(_ => cancellation.Cancel());
            int calls = 0;
            Assert.Throws<OperationCanceledException>(() => KimodoMotionSequence.Generate(request, (e, c, g, t) =>
            {
                calls++;
                return Generate(e, c, g, t);
            }, cancellation.Token));
            Assert.AreEqual(1, calls);
        }

        [Test]
        public void Duration_RoundsToFramesAndRejectsInvalidValues()
        {
            Assert.AreEqual(37, new CharacterMotionPart(null, durationSeconds: 1.234f).FrameCount);
            foreach (float duration in new[] { 0f, -1f, float.NaN, float.PositiveInfinity })
                Assert.Throws<ArgumentOutOfRangeException>(() => _ = new CharacterMotionPart(null, durationSeconds: duration).FrameCount);
        }

        private static CharacterMotionRequest Request(int frames) => new CharacterMotionRequest
        {
            Segments = new[] { new CharacterMotionSegment(new KimodoTextEmbedding(new float[4096]), new KimodoGenerationRequest(frameCount: frames)) }
        };

        [Test]
        public void Correction_RotatesPoseToConstrainedHeading()
        {
            var raw = Generate(null, KimodoConditioning.CreateEmpty(2), new KimodoGenerationRequest(frameCount: 2), default);
            var constraints = new KimodoConstraintSet(new KimodoRootConstraint(new[] { 0 },
                new[] { Vector2.zero }, new[] { new Vector2(0, 1) }));
            KimodoMotionProcessing.Correct(raw, new KimodoConstraintCompiler().Compile(constraints, 2), default);
            Assert.Less(Quaternion.Angle(KimodoMotionProcessing.Rotation(raw, 95), Quaternion.Euler(0, 90, 0)), .01f);
            Assert.That(KimodoMotionProcessing.Value(raw, 4), Is.EqualTo(1f).Within(1e-5f));
        }

        [Test]
        public void Correction_PreservesFeasibleFullPoseAndHistoryTransform()
        {
            var raw = Generate(null, KimodoConditioning.CreateEmpty(2), new KimodoGenerationRequest(frameCount: 2), default);
            var expected = KimodoMotionProcessing.TransformHistory(raw, new Vector3(2, .2f, 3), Quaternion.Euler(0, 45, 0));
            var positions = new Vector3[30];
            for (int j = 0; j < 30; j++) positions[j] = Position(expected, 0, j);
            var conditioning = new KimodoConstraintCompiler().Compile(new KimodoConstraintSet(
                new KimodoFullBodyConstraint(new[] { 0 }, positions)), 2);
            KimodoMotionProcessing.Correct(raw, conditioning, default);
            for (int j = 0; j < 30; j++) Assert.Less(Vector3.Distance(Position(raw, 0, j), positions[j]), .001f, $"Joint {j}");
        }

        private static Vector3 Position(float[] raw, int frame, int joint)
        {
            int offset = frame * 369;
            return new Vector3(KimodoMotionProcessing.Value(raw, offset) + KimodoMotionProcessing.Value(raw, offset + 5 + joint * 3),
                KimodoMotionProcessing.Value(raw, offset + 6 + joint * 3),
                KimodoMotionProcessing.Value(raw, offset + 2) + KimodoMotionProcessing.Value(raw, offset + 7 + joint * 3));
        }

        [Test]
        public void Correction_KeepsContactingFeetPlantedWhilePelvisMoves()
        {
            const int frames = 10;
            var raw = Generate(null, KimodoConditioning.CreateEmpty(frames), new KimodoGenerationRequest(frameCount: frames), default);
            Vector3 left = Position(raw, 0, 24), right = Position(raw, 0, 28);
            for (int frame = 0; frame < frames; frame++)
            {
                int offset = frame * 369;
                raw[offset] += frame * .0005f / KimodoSomaRuntimeData.Scale[0];
                raw[offset + 6] -= frame * .001f / KimodoSomaRuntimeData.Scale[6];
                for (int contact = 365; contact < 369; contact++)
                    raw[offset + contact] = (1f - KimodoSomaRuntimeData.Mean[contact]) / KimodoSomaRuntimeData.Scale[contact];
            }
            KimodoMotionProcessing.Correct(raw, KimodoConditioning.CreateEmpty(frames), default);
            for (int frame = 0; frame < frames; frame++)
            {
                Assert.Less(Vector3.Distance(Position(raw, frame, 24), left), .0001f);
                Assert.Less(Vector3.Distance(Position(raw, frame, 28), right), .0001f);
            }
        }

        private static float[] Generate(KimodoTextEmbedding embedding, KimodoConditioning conditioning,
            KimodoGenerationRequest generation, CancellationToken token)
        {
            int frames = generation.FrameCount;
            var features = new float[frames * 369];
            var joints = new Vector3[30];
            joints[0] = Vector3.up;
            for (int j = 1; j < 30; j++) joints[j] = joints[KimodoSomaRuntimeData.Parents[j]] + KimodoSomaRuntimeData.RestOffsets[j];
            for (int f = 0; f < frames; f++)
            {
                int offset = f * 369;
                features[offset + 1] = 1f;
                features[offset + 3] = 1f;
                for (int j = 0; j < 30; j++)
                {
                    features[offset + 5 + j * 3] = joints[j].x;
                    features[offset + 6 + j * 3] = joints[j].y;
                    features[offset + 7 + j * 3] = joints[j].z;
                    features[offset + 95 + j * 6] = 1f;
                    features[offset + 99 + j * 6] = 1f;
                }
                for (int k = 0; k < 369; k++)
                    features[offset + k] = conditioning.MotionMask.Span[offset + k]
                        ? conditioning.ObservedMotion.Span[offset + k]
                        : (features[offset + k] - KimodoSomaRuntimeData.Mean[k]) / KimodoSomaRuntimeData.Scale[k];
            }
            return features;
        }

        private sealed class ImmediateProgress : IProgress<int>
        {
            private readonly Action<int> report;
            public ImmediateProgress(Action<int> report) => this.report = report;
            public void Report(int value) => report(value);
        }
    }
}
