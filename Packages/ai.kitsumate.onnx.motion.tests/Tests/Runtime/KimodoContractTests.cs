using System;
using KitsuMate.Onnx.Motion.Kimodo;
using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Tests
{
    public class KimodoContractTests
    {
        [Test]
        public void TextEmbedding_RequiresExactly4096FiniteValues()
        {
            Assert.Throws<ArgumentException>(() => new KimodoTextEmbedding(new float[12]));
            var invalid = new float[KimodoTextEmbedding.Dimension];
            invalid[17] = float.NaN;
            Assert.Throws<ArgumentException>(() => new KimodoTextEmbedding(invalid));

            var valid = new KimodoTextEmbedding(new float[KimodoTextEmbedding.Dimension], "reference");
            Assert.AreEqual(KimodoTextEmbedding.Dimension, valid.Values.Length);
            Assert.AreEqual("reference", valid.ModelId);
        }

        [Test]
        public void DiffusionSchedule_MapsEndpointsAndFinalStepReturnsCleanPrediction()
        {
            var schedule = new KimodoDiffusionSchedule(100);
            Assert.AreEqual(0, schedule.GetModelTimestep(0));
            Assert.AreEqual(999, schedule.GetModelTimestep(99));

            var current = new[] { 10f, -3f };
            var clean = new[] { 2f, 5f };
            var destination = new float[2];
            schedule.Step(current, clean, 0, destination);
            Assert.AreEqual(clean[0], destination[0], 1e-5f);
            Assert.AreEqual(clean[1], destination[1], 1e-5f);
        }

        [Test]
        public void Decoder_ProducesFiniteNormalizedHumanoidRotations()
        {
            var normalized = new float[KimodoTensorContract.Frames * KimodoTensorContract.MotionDimension];
            var motion = KimodoMotionDecoder.Decode(normalized, KimodoTensorContract.Frames, null);

            Assert.AreEqual(60, motion.FrameCount);
            Assert.AreEqual(30f, motion.FramesPerSecond);
            Assert.IsTrue(motion.BoneAvailability[(int)HumanBodyBones.Hips]);
            Assert.IsTrue(motion.BoneAvailability[(int)HumanBodyBones.LeftFoot]);

            for (int frame = 0; frame < motion.FrameCount; frame++)
            {
                Vector3 root = motion.RootPositions[frame];
                Assert.IsTrue(IsFinite(root.x) && IsFinite(root.y) && IsFinite(root.z));
                Quaternion hips = motion.GetBoneRotation(frame, HumanBodyBones.Hips);
                float norm = Mathf.Sqrt(hips.x * hips.x + hips.y * hips.y + hips.z * hips.z + hips.w * hips.w);
                Assert.AreEqual(1f, norm, 1e-3f);
            }
        }

        private static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
    }
}
