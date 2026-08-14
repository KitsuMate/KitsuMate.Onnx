using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using KitsuMate.Onnx;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace KitsuMate.Onnx.Motion.Tests.PlayMode
{
    public sealed class KimodoConstraintPlayModeTests
    {
        private const float RootHeightFraction = 0.02f;
        private const float EffectorLimbFraction = 0.05f;
        private const float HeadingToleranceDegrees = 5f;

        [UnityTest]
        [Explicit("Loads the installed Kimodo model and can take several minutes.")]
        [Category("LargeModel")]
        [Timeout(600000)]
        public IEnumerator GeneratedMotion_RespectsAuthoredRootAndEffectorConstraints()
        {
            OnnxSettings settings = OnnxSettings.Load();
            if (settings == null) Assert.Ignore("Create a Resources/OnnxSettings asset to run the Kimodo integration test.");
            CharacterMotionEngine engine = settings.GetDefaultEngine<CharacterMotionEngine>();
            if (engine == null) Assert.Ignore("Register a default CharacterMotionEngine in OnnxSettings.");
            if (engine.Backend == null) Assert.Ignore("Assign an Auto-capable backend to the CharacterMotionEngine.");

            GameObject prefab = Resources.Load<GameObject>("XBot");
            TextAsset embeddingBytes = Resources.Load<TextAsset>("Llm2VecReference");
            if (prefab == null) Assert.Ignore("The XBot Humanoid Play Mode fixture is unavailable.");
            if (embeddingBytes == null) Assert.Ignore("The prebaked LLM2Vec Play Mode fixture is unavailable.");

            GameObject character = null;
            GameObject authoringRoot = null;
            InferenceEngineRuntime<CharacterMotionRequest, CharacterMotionResult> runtime = null;
            AnimationClip humanoidClip = null;
            try
            {
                character = UnityEngine.Object.Instantiate(prefab);
                character.name = "KimodoConstraintTest_XBot";
                Animator animator = character.GetComponentInChildren<Animator>(true);
                Assert.That(animator, Is.Not.Null, "XBot must contain an Animator.");
                Assert.That(animator.avatar, Is.Not.Null);
                Assert.That(animator.avatar.isHuman, Is.True, "XBot must import as Humanoid.");
                animator.Rebind();
                animator.Update(0f);
                animator.enabled = false;

                authoringRoot = new GameObject("KimodoConstraintTest_Motion");
                CharacterMotion motion = authoringRoot.AddComponent<CharacterMotion>();
                motion.ConfigureTarget(animator, authoringRoot.transform);
                motion.ConfigureInference(engine);

                Quaternion[] bindRotations = CaptureLocalRotations(animator);
                CaptureGlobalRetargetData(animator, out Quaternion[] bindGlobals,
                    out Quaternion[] bindParentGlobals, out int[] humanoidParents);
                Vector3 bindHipsOffset = animator.transform.InverseTransformPoint(animator.GetBoneTransform(HumanBodyBones.Hips).position);

                Vector3 avatarForward = AvatarForward(animator);
                Quaternion constraintRotation = Quaternion.LookRotation(avatarForward, Vector3.up);

                CharacterMotionKeyframe start = CreateKeyframe(motion, animator, 0, Vector3.zero, constraintRotation,
                    CharacterMotionConstraintType.RootPosition | CharacterMotionConstraintType.RootHeading);
                CharacterMotionKeyframe middle = CreateKeyframe(motion, animator, 30, avatarForward, constraintRotation,
                    CharacterMotionConstraintType.RootPosition | CharacterMotionConstraintType.RootHeading | CharacterMotionConstraintType.RightHand);
                CharacterMotionKeyframe end = CreateKeyframe(motion, animator, 59, avatarForward * 2f, constraintRotation,
                    CharacterMotionConstraintType.RootPosition | CharacterMotionConstraintType.RootHeading | CharacterMotionConstraintType.LeftFoot);

                Transform rightHandTarget = CreateEffectorTarget(middle, animator, HumanBodyBones.RightHand, "RightHand");
                rightHandTarget.position += middle.transform.forward * 0.15f;
                middle.SetEffector(CharacterMotionConstraintType.RightHand, rightHandTarget);
                Transform leftFootTarget = CreateEffectorTarget(end, animator, HumanBodyBones.LeftFoot, "LeftFoot");
                end.SetEffector(CharacterMotionConstraintType.LeftFoot, leftFootTarget);

                CharacterMotionValidationResult validation = motion.ValidateMotion(requireIntent: false);
                Assert.That(validation.IsValid, Is.True, string.Join("\n", validation.Diagnostics));

                KimodoTextEmbedding embedding = ReadEmbedding(embeddingBytes.bytes);
                Vector2 firstHeading = CharacterMotionSpace.WorldForwardToHeading(start.transform.forward, motion.ActionOrigin);
                var intent = new TestIntent(Mathf.Atan2(firstHeading.y, firstHeading.x));
                CharacterMotionRequest request = motion.BuildEngineRequest(embedding, intent);

                Task<InferenceEngineRuntime<CharacterMotionRequest, CharacterMotionResult>> load = engine.CreateRuntimeAsync();
                while (!load.IsCompleted) yield return null;
                ThrowIfFailed(load);
                runtime = load.Result;

                Task<CharacterMotionResult> generation = runtime.RunAsync(request);
                while (!generation.IsCompleted) yield return null;
                ThrowIfFailed(generation);
                KimodoHumanoidMotion generated = generation.Result.Motion;

                AssertValidOutput(generated);
                humanoidClip = BuildHumanoidClip(animator, generated, bindRotations, bindGlobals,
                    bindParentGlobals, humanoidParents, bindHipsOffset, motion.ActionOrigin);
                Assert.That(humanoidClip.humanMotion, Is.True, "The generated test clip must use Unity Humanoid curves.");
                animator.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
                for (int i = 0; i < bindRotations.Length; i++)
                {
                    Transform bone = animator.GetBoneTransform((HumanBodyBones)i);
                    if (bone != null) bone.localRotation = bindRotations[i];
                }

                float avatarHeight = AvatarHeight(animator);
                float rootTolerance = avatarHeight * RootHeightFraction;
                float armTolerance = LimbLength(animator, HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand) * EffectorLimbFraction;
                float legTolerance = LimbLength(animator, HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot) * EffectorLimbFraction;

                var violations = new List<string>();
                CheckFrameConstraints(animator, humanoidClip, start, null, 0f, rootTolerance, violations);
                CheckFrameConstraints(animator, humanoidClip, middle, rightHandTarget, armTolerance, rootTolerance, violations);
                CheckFrameConstraints(animator, humanoidClip, end, leftFootTarget, legTolerance, rootTolerance, violations);
                Assert.That(violations, Is.Empty, string.Join("\n", violations));
            }
            finally
            {
                runtime?.Dispose();
                if (humanoidClip != null) UnityEngine.Object.Destroy(humanoidClip);
                if (authoringRoot != null) UnityEngine.Object.Destroy(authoringRoot);
                if (character != null) UnityEngine.Object.Destroy(character);
            }

            yield return null;
        }

        private static CharacterMotionKeyframe CreateKeyframe(CharacterMotion motion, Animator animator, int frame, Vector3 position, Quaternion rotation, CharacterMotionConstraintType constraints)
        {
            var keyframeObject = new GameObject($"Frame_{frame:00}");
            keyframeObject.transform.SetParent(motion.transform, false);
            keyframeObject.transform.SetPositionAndRotation(position, rotation);
            CharacterMotionKeyframe keyframe = keyframeObject.AddComponent<CharacterMotionKeyframe>();
            keyframe.Configure(frame, constraints);

            if (keyframe.RequiresPose)
            {
                Vector3 savedPosition = animator.transform.position;
                animator.transform.position = keyframe.transform.position;
                keyframe.Pose.Capture(animator, keyframe.transform);
                animator.transform.position = savedPosition;
            }

            return keyframe;
        }

        private static Transform CreateEffectorTarget(CharacterMotionKeyframe keyframe, Animator animator, HumanBodyBones bone, string name)
        {
            Vector3 savedPosition = animator.transform.position;
            animator.transform.position = keyframe.transform.position;
            Transform source = animator.GetBoneTransform(bone);
            var target = new GameObject(name);
            target.transform.SetParent(keyframe.transform, true);
            target.transform.SetPositionAndRotation(source.position, source.rotation);
            animator.transform.position = savedPosition;
            return target.transform;
        }

        private static Vector3 AvatarForward(Animator animator)
        {
            Vector3 left = animator.GetBoneTransform(HumanBodyBones.LeftUpperLeg).position;
            Vector3 right = animator.GetBoneTransform(HumanBodyBones.RightUpperLeg).position;
            Vector3 forward = Vector3.Cross(right - left, Vector3.up);
            forward = Vector3.ProjectOnPlane(forward, Vector3.up).normalized;
            Assert.That(forward.sqrMagnitude, Is.GreaterThan(0.5f), "XBot's hips do not define a stable horizontal heading.");
            return forward;
        }

        private static void CheckFrameConstraints(
            Animator animator,
            AnimationClip clip,
            CharacterMotionKeyframe keyframe,
            Transform effectorTarget,
            float effectorTolerance,
            float rootTolerance,
            List<string> violations)
        {
            clip.SampleAnimation(animator.gameObject, keyframe.Frame / clip.frameRate);

            using var poseHandler = new HumanPoseHandler(animator.avatar, animator.transform);
            var humanPose = new HumanPose();
            poseHandler.GetHumanPose(ref humanPose);

            // Root constraints address Kimodo's trajectory/body-root feature, not the Hips transform.
            // The Hips bone also contains the model's local gait displacement and is therefore not a
            // faithful measurement of the authored root control after Humanoid retargeting.
            Vector3 generatedRootWorld = humanPose.bodyPosition;
            Vector2 generatedRoot = new Vector2(generatedRootWorld.x, generatedRootWorld.z);
            Vector2 targetRoot = new Vector2(keyframe.transform.position.x, keyframe.transform.position.z);
            float rootError = Vector2.Distance(generatedRoot, targetRoot);
            TestContext.WriteLine($"Frame {keyframe.Frame} root error={rootError:F4} m, tolerance={rootTolerance:F4} m.");
            if (rootError > rootTolerance)
                violations.Add($"Frame {keyframe.Frame} root constraint failed. Target={targetRoot}, generated={generatedRoot}, error={rootError:F4}, tolerance={rootTolerance:F4}.");

            Vector3 generatedHeading = Vector3.ProjectOnPlane(humanPose.bodyRotation * Vector3.forward, Vector3.up).normalized;
            Vector3 targetHeading = Vector3.ProjectOnPlane(keyframe.transform.forward, Vector3.up).normalized;
            float headingError = Vector3.Angle(generatedHeading, targetHeading);
            TestContext.WriteLine($"Frame {keyframe.Frame} heading error={headingError:F3} degrees, tolerance={HeadingToleranceDegrees:F3} degrees.");
            if (headingError > HeadingToleranceDegrees)
                violations.Add($"Frame {keyframe.Frame} heading constraint failed. Target={targetHeading}, generated={generatedHeading}, error={headingError:F3} degrees, tolerance={HeadingToleranceDegrees:F3} degrees.");

            if (effectorTarget == null) return;
            HumanBodyBones bone = (keyframe.Constraints & CharacterMotionConstraintType.RightHand) != 0
                ? HumanBodyBones.RightHand
                : HumanBodyBones.LeftFoot;
            Vector3 generatedEffector = animator.GetBoneTransform(bone).position;
            float effectorError = Vector3.Distance(generatedEffector, effectorTarget.position);
            TestContext.WriteLine($"Frame {keyframe.Frame} {bone} error={effectorError:F4} m, tolerance={effectorTolerance:F4} m.");
            if (effectorError > effectorTolerance)
                violations.Add($"Frame {keyframe.Frame} {bone} constraint failed. Target={effectorTarget.position}, generated={generatedEffector}, error={effectorError:F4}, tolerance={effectorTolerance:F4}.");
        }

        private static void ApplyFrame(
            Animator animator,
            KimodoHumanoidMotion motion,
            int frame,
            Quaternion[] bindRotations,
            Quaternion[] bindGlobals,
            Quaternion[] bindParentGlobals,
            int[] humanoidParents,
            Vector3 bindHipsOffset,
            Transform actionOrigin)
        {
            animator.transform.rotation = Quaternion.Normalize(actionOrigin.rotation * motion.RootRotations[frame]);
            Vector3 generatedHips = actionOrigin.TransformPoint(motion.RootPositions[frame]);
            animator.transform.position = generatedHips - animator.transform.rotation * bindHipsOffset;
            for (int i = 0; i < bindRotations.Length; i++)
            {
                if (!motion.BoneAvailability[i]) continue;
                Transform bone = animator.GetBoneTransform((HumanBodyBones)i);
                if (bone == null) continue;
                if (motion.BoneGlobalRotationDeltas == null)
                {
                    bone.localRotation = Quaternion.Normalize(bindRotations[i] * motion.GetBoneRotation(frame, (HumanBodyBones)i));
                    continue;
                }
                int parent = humanoidParents[i];
                while (parent >= 0 && !motion.BoneAvailability[parent]) parent = humanoidParents[parent];
                Quaternion parentDelta = parent >= 0
                    ? motion.GetBoneGlobalRotation(frame, (HumanBodyBones)parent)
                    : Quaternion.identity;
                Quaternion boneDelta = motion.GetBoneGlobalRotation(frame, (HumanBodyBones)i);
                bone.localRotation = Quaternion.Normalize(
                    Quaternion.Inverse(parentDelta * bindParentGlobals[i]) * (boneDelta * bindGlobals[i]));
            }
        }

        private static AnimationClip BuildHumanoidClip(
            Animator animator,
            KimodoHumanoidMotion motion,
            Quaternion[] bindRotations,
            Quaternion[] bindGlobals,
            Quaternion[] bindParentGlobals,
            int[] humanoidParents,
            Vector3 bindHipsOffset,
            Transform actionOrigin)
        {
            int frames = motion.FrameCount;
            int muscleCount = HumanTrait.MuscleCount;
            var bodyPositions = new Vector3[frames];
            var bodyRotations = new Quaternion[frames];
            var muscles = new float[muscleCount][];
            for (int muscle = 0; muscle < muscleCount; muscle++) muscles[muscle] = new float[frames];

            using (var handler = new HumanPoseHandler(animator.avatar, animator.transform))
            {
                var pose = new HumanPose();
                for (int frame = 0; frame < frames; frame++)
                {
                    ApplyFrame(animator, motion, frame, bindRotations, bindGlobals, bindParentGlobals,
                        humanoidParents, bindHipsOffset, actionOrigin);
                    handler.GetHumanPose(ref pose);
                    bodyPositions[frame] = pose.bodyPosition;
                    bodyRotations[frame] = pose.bodyRotation.normalized;
                    if (frame > 0 && Quaternion.Dot(bodyRotations[frame - 1], bodyRotations[frame]) < 0f)
                    {
                        Quaternion q = bodyRotations[frame];
                        bodyRotations[frame] = new Quaternion(-q.x, -q.y, -q.z, -q.w);
                    }
                    for (int muscle = 0; muscle < muscleCount; muscle++) muscles[muscle][frame] = pose.muscles[muscle];
                }
            }

            var clip = new AnimationClip
            {
                name = "KimodoConstraintHumanoidTest",
                frameRate = motion.FramesPerSecond,
                wrapMode = WrapMode.ClampForever,
            };
            WriteVector("RootT", bodyPositions);
            WriteQuaternion("RootQ", bodyRotations);
            for (int muscle = 0; muscle < muscleCount; muscle++)
                WriteCurve(HumanTrait.MuscleName[muscle], muscles[muscle]);
            clip.EnsureQuaternionContinuity();
            return clip;

            void WriteVector(string prefix, Vector3[] values)
            {
                WriteCurve(prefix + ".x", Array.ConvertAll(values, value => value.x));
                WriteCurve(prefix + ".y", Array.ConvertAll(values, value => value.y));
                WriteCurve(prefix + ".z", Array.ConvertAll(values, value => value.z));
            }

            void WriteQuaternion(string prefix, Quaternion[] values)
            {
                WriteCurve(prefix + ".x", Array.ConvertAll(values, value => value.x));
                WriteCurve(prefix + ".y", Array.ConvertAll(values, value => value.y));
                WriteCurve(prefix + ".z", Array.ConvertAll(values, value => value.z));
                WriteCurve(prefix + ".w", Array.ConvertAll(values, value => value.w));
            }

            void WriteCurve(string property, float[] values)
            {
                var keys = new Keyframe[values.Length + 1];
                for (int i = 0; i <= values.Length; i++)
                    keys[i] = new Keyframe(i / motion.FramesPerSecond, values[Math.Min(i, values.Length - 1)]);
                clip.SetCurve(string.Empty, typeof(Animator), property, new AnimationCurve(keys));
            }
        }

        private static void CaptureGlobalRetargetData(
            Animator animator,
            out Quaternion[] bindGlobals,
            out Quaternion[] bindParentGlobals,
            out int[] humanoidParents)
        {
            int count = (int)HumanBodyBones.LastBone;
            bindGlobals = new Quaternion[count];
            bindParentGlobals = new Quaternion[count];
            humanoidParents = new int[count];
            Array.Fill(humanoidParents, -1);
            Quaternion inverseRoot = Quaternion.Inverse(animator.transform.rotation);
            var transformToBone = new Dictionary<Transform, int>();
            for (int i = 0; i < count; i++)
            {
                Transform bone = animator.GetBoneTransform((HumanBodyBones)i);
                if (bone != null) transformToBone[bone] = i;
            }
            for (int i = 0; i < count; i++)
            {
                Transform bone = animator.GetBoneTransform((HumanBodyBones)i);
                if (bone == null)
                {
                    bindGlobals[i] = bindParentGlobals[i] = Quaternion.identity;
                    continue;
                }
                bindGlobals[i] = inverseRoot * bone.rotation;
                Transform parent = bone.parent;
                bindParentGlobals[i] = parent == null || parent == animator.transform
                    ? Quaternion.identity
                    : inverseRoot * parent.rotation;
                for (Transform ancestor = parent; ancestor != null && ancestor != animator.transform; ancestor = ancestor.parent)
                    if (transformToBone.TryGetValue(ancestor, out int parentIndex))
                    {
                        humanoidParents[i] = parentIndex;
                        break;
                    }
            }
        }

        private static Quaternion[] CaptureLocalRotations(Animator animator)
        {
            int count = (int)HumanBodyBones.LastBone;
            var result = new Quaternion[count];
            for (int i = 0; i < count; i++)
            {
                Transform bone = animator.GetBoneTransform((HumanBodyBones)i);
                result[i] = bone != null ? bone.localRotation : Quaternion.identity;
            }
            return result;
        }

        private static KimodoTextEmbedding ReadEmbedding(byte[] bytes)
        {
            Assert.That(bytes.Length, Is.EqualTo(KimodoTextEmbedding.Dimension * sizeof(float)));
            var values = new float[KimodoTextEmbedding.Dimension];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return new KimodoTextEmbedding(values, "playmode-reference", copy: false);
        }

        private static void AssertValidOutput(KimodoHumanoidMotion motion)
        {
            Assert.That(motion, Is.Not.Null);
            Assert.That(motion.FrameCount, Is.EqualTo(KimodoConditioning.DefaultFrameCount));
            Assert.That(motion.SmoothedRootPositions, Has.Length.EqualTo(motion.FrameCount));
            foreach (Vector3 position in motion.RootPositions)
                Assert.That(Finite(position.x) && Finite(position.y) && Finite(position.z), Is.True, "Generated root contains a non-finite value.");
            foreach (Quaternion rotation in motion.BoneRotationDeltas)
                Assert.That(Finite(rotation.x) && Finite(rotation.y) && Finite(rotation.z) && Finite(rotation.w), Is.True, "Generated pose contains a non-finite value.");
        }

        private static float AvatarHeight(Animator animator)
        {
            Transform head = animator.GetBoneTransform(HumanBodyBones.Head);
            Transform leftFoot = animator.GetBoneTransform(HumanBodyBones.LeftFoot);
            Transform rightFoot = animator.GetBoneTransform(HumanBodyBones.RightFoot);
            return head.position.y - Mathf.Min(leftFoot.position.y, rightFoot.position.y);
        }

        private static float LimbLength(Animator animator, HumanBodyBones upper, HumanBodyBones lower, HumanBodyBones end)
        {
            Transform a = animator.GetBoneTransform(upper), b = animator.GetBoneTransform(lower), c = animator.GetBoneTransform(end);
            return Vector3.Distance(a.position, b.position) + Vector3.Distance(b.position, c.position);
        }

        private static void ThrowIfFailed(Task task)
        {
            if (task.IsCanceled) throw new OperationCanceledException("Kimodo integration operation was canceled.");
            if (task.IsFaulted) throw task.Exception?.InnerException ?? task.Exception;
        }

        private static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

        private sealed class TestIntent : ICharacterMotionIntent
        {
            private readonly float firstHeadingRadians;

            public TestIntent(float firstHeadingRadians) => this.firstHeadingRadians = firstHeadingRadians;
            public string Prompt => "A person walks forward and waves their right hand.";
            public CharacterMotionGenerationSettings Settings => new CharacterMotionGenerationSettings
            {
                DenoisingSteps = 25,
                Seed = 0,
                TextGuidance = 2f,
                ConstraintGuidance = 4f,
                FirstHeadingRadians = firstHeadingRadians,
            };

            public bool TryGetEmbedding(ModelIdentity encoder, out KimodoTextEmbedding embedding)
            {
                embedding = null;
                return false;
            }
        }
    }
}
