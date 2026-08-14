using System;
using System.Collections.Generic;
using System.Linq;
using KitsuMate.Onnx.Motion.Kimodo;
using KitsuMate.Onnx.Motion.Editor;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Tests
{
    public sealed class CharacterMotionAuthoringTests
    {
        [Test]
        public void Planner_ExpandsRepetitionsAndCalculatesOverlappingTimeline()
        {
            CharacterMotionIntent walk = ScriptableObject.CreateInstance<CharacterMotionIntent>();
            CharacterMotionIntent sit = ScriptableObject.CreateInstance<CharacterMotionIntent>();
            try
            {
                var parts = new[]
                {
                    new CharacterMotionPart(walk, 2),
                    new CharacterMotionPart(sit),
                };

                CharacterMotionGenerationPlan plan = CharacterMotionPlanner.Build(parts, 5, 7, hasPreviousMotion: true);

                Assert.AreEqual(3, plan.Runs.Length);
                Assert.AreEqual(170, plan.OutputFrameCount);
                Assert.AreEqual(0, plan.Runs[0].OutputStartFrame);
                Assert.AreEqual(55, plan.Runs[1].OutputStartFrame);
                Assert.AreEqual(110, plan.Runs[2].OutputStartFrame);
                Assert.AreEqual(7, plan.Runs[0].LeadingOverlapFrames);
                Assert.AreEqual(5, plan.Runs[1].LeadingOverlapFrames);
                Assert.AreEqual(5, plan.Runs[2].LeadingOverlapFrames);
                Assert.AreSame(walk, plan.Runs[0].Intent);
                Assert.AreSame(walk, plan.Runs[1].Intent);
                Assert.AreSame(sit, plan.Runs[2].Intent);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(walk);
                UnityEngine.Object.DestroyImmediate(sit);
            }
        }

        [Test]
        public void Motion_SinglePartKeepsExistingSixtyFrameContract()
        {
            var gameObject = new GameObject("SinglePartMotion");
            var previousObject = new GameObject("PreviousMotion");
            CharacterMotionIntent intent = ScriptableObject.CreateInstance<CharacterMotionIntent>();
            try
            {
                CharacterMotion motion = gameObject.AddComponent<CharacterMotion>();
                CharacterMotion previous = previousObject.AddComponent<CharacterMotion>();
                motion.SetParts(new[] { new CharacterMotionPart(intent) });

                Assert.AreSame(intent, motion.Intent);
                Assert.AreEqual(1, motion.Timeline.RunCount);
                Assert.AreEqual(60, motion.Timeline.FrameCount);
                Assert.AreEqual(60, motion.Timeline.EffectiveChainedFrameCount);

                motion.ConfigurePrevious(previous, 5);
                Assert.AreEqual(60, motion.Timeline.FrameCount,
                    "Entry overlap stays in the standalone clip.");
                Assert.AreEqual(55, motion.Timeline.EffectiveChainedFrameCount,
                    "Chained playback overlaps the configured entry frames.");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(gameObject);
                UnityEngine.Object.DestroyImmediate(previousObject);
                UnityEngine.Object.DestroyImmediate(intent);
            }
        }

        [Test]
        public void GenerationSettings_AreHiddenAndIgnoredWhenOverrideIsDisabled()
        {
            CharacterMotionIntent intent = ScriptableObject.CreateInstance<CharacterMotionIntent>();
            var gameObject = new GameObject("SettingsDrawerMotion");
            try
            {
                var intentObject = new SerializedObject(intent);
                intentObject.FindProperty("useCustomGenerationSettings").boolValue = false;
                SerializedProperty intentSettings = intentObject.FindProperty("settings");
                intentSettings.FindPropertyRelative("DenoisingSteps").intValue = 40;
                intentSettings.FindPropertyRelative("TextGuidance").floatValue = 9f;
                intentObject.ApplyModifiedPropertiesWithoutUndo();
                Assert.AreEqual(CharacterMotionGenerationSettings.Default.DenoisingSteps, intent.Settings.DenoisingSteps);
                Assert.AreEqual(CharacterMotionGenerationSettings.Default.TextGuidance, intent.Settings.TextGuidance);

                CharacterMotion motion = gameObject.AddComponent<CharacterMotion>();
                motion.SetParts(new[] { new CharacterMotionPart(intent) });
                var serialized = new SerializedObject(motion);
                SerializedProperty part = serialized.FindProperty("parts").GetArrayElementAtIndex(0);
                part.isExpanded = true;
                SerializedProperty enabled = part.FindPropertyRelative("overrideGenerationSettings");
                var drawer = new CharacterMotionPartDrawer();
                enabled.boolValue = false;
                float collapsedSettingsHeight = drawer.GetPropertyHeight(part, new GUIContent("Part"));
                enabled.boolValue = true;
                float expandedSettingsHeight = drawer.GetPropertyHeight(part, new GUIContent("Part"));
                Assert.Greater(expandedSettingsHeight, collapsedSettingsHeight);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(gameObject);
                UnityEngine.Object.DestroyImmediate(intent);
            }
        }

        [Test]
        public void Intent_EmbeddingRequiresMatchingPromptAndEncoder()
        {
            CharacterMotionIntent intent = ScriptableObject.CreateInstance<CharacterMotionIntent>();
            try
            {
                SetPrompt(intent, "A person waves.");
                var identity = new ModelIdentity("embeddings", "encoder", "v1", "fp32:hash");
                var values = new float[KimodoTextEmbedding.Dimension];
                values[17] = 0.5f;
                intent.SetEmbedding(identity, new KimodoTextEmbedding(values));
                Assert.IsTrue(intent.TryGetEmbedding(identity, out KimodoTextEmbedding embedding));
                Assert.AreEqual(0.5f, embedding.Values.Span[17]);
                Assert.IsFalse(intent.TryGetEmbedding(new ModelIdentity("embeddings", "encoder", "v2", "fp32:hash"), out _));
                SetPrompt(intent, "A person sits.");
                Assert.IsFalse(intent.TryGetEmbedding(identity, out _));
            }
            finally { UnityEngine.Object.DestroyImmediate(intent); }
        }

        [Test]
        public void PoseSnapshot_CopiesDataAndTracksAvatar()
        {
            int bones = (int)HumanBodyBones.LastBone;
            var humanoid = Enumerable.Repeat(Quaternion.identity, bones).ToArray();
            var available = Enumerable.Repeat(true, bones).ToArray();
            var positions = Enumerable.Range(0, 30).Select(i => new Vector3(i, i * 2f, -i)).ToArray();
            var rotations = Enumerable.Repeat(Quaternion.identity, 30).ToArray();
            var snapshot = new CharacterMotionPoseSnapshot();
            snapshot.Set(humanoid, available, positions, rotations, "avatar-a");
            positions[5] = Vector3.zero;
            Assert.IsTrue(snapshot.IsCaptured);
            Assert.IsTrue(snapshot.MatchesAvatar("avatar-a"));
            Assert.IsFalse(snapshot.MatchesAvatar("avatar-b"));
            Assert.AreEqual(new Vector3(5f, 10f, -5f), snapshot.SomaPositionsRelativeToRoot.Span[5]);
        }

        [Test]
        public void CharacterMotionSpace_ConvertsRelativeCoordinatesAndHeading()
        {
            var originObject = new GameObject("origin");
            try
            {
                originObject.transform.SetPositionAndRotation(new Vector3(10f, 1f, 20f), Quaternion.Euler(0f, 90f, 0f));
                Vector3 world = originObject.transform.position + originObject.transform.rotation * new Vector3(1f, 2f, 3f);
                AssertVector(new Vector3(-1f, 2f, 3f), CharacterMotionSpace.WorldToCanonical(world, originObject.transform));
                Vector2 heading = CharacterMotionSpace.WorldForwardToHeading(originObject.transform.rotation * Vector3.right, originObject.transform);
                Assert.AreEqual(0f, heading.x, 1e-5f);
                Assert.AreEqual(-1f, heading.y, 1e-5f);
                Quaternion roundTrip = CharacterMotionSpace.ReflectX(CharacterMotionSpace.ReflectX(Quaternion.Euler(15f, 25f, -35f)));
                Assert.Less(Quaternion.Angle(Quaternion.Euler(15f, 25f, -35f), roundTrip), 1e-3f);
            }
            finally { UnityEngine.Object.DestroyImmediate(originObject); }
        }

        [Test]
        public void HumanoidSomaMapper_UsesDirectBonesAndDocumentedFallbacks()
        {
            var created = new List<GameObject>();
            var bones = new Dictionary<HumanBodyBones, Transform>();
            try
            {
                Transform Add(HumanBodyBones bone, Vector3 position)
                {
                    var go = new GameObject(bone.ToString()); created.Add(go); go.transform.position = position; bones[bone] = go.transform; return go.transform;
                }
                Add(HumanBodyBones.Hips, new Vector3(0f, 1f, 0f));
                Add(HumanBodyBones.Spine, new Vector3(0f, 1.2f, 0f));
                Add(HumanBodyBones.Chest, new Vector3(0f, 1.4f, 0f));
                Add(HumanBodyBones.Neck, new Vector3(0f, 1.65f, 0f));
                Add(HumanBodyBones.Head, new Vector3(0f, 1.8f, 0f));
                AddArm(true); AddArm(false); AddLeg(true); AddLeg(false);

                var diagnostics = new CharacterMotionValidationResult();
                bool success = HumanoidSomaMapper.TryCapture(bone => bones.TryGetValue(bone, out Transform value) ? value : null,
                    out Vector3[] positions, out Quaternion[] rotations, diagnostics);
                Assert.IsTrue(success, string.Join("\n", diagnostics.Diagnostics));
                Assert.AreEqual(bones[HumanBodyBones.LeftHand].position, positions[(int)KimodoJoint.LeftHand]);
                Assert.Greater(Vector3.Distance(positions[(int)KimodoJoint.LeftHandMiddleEnd], positions[(int)KimodoJoint.LeftHand]), 0.01f);
                Assert.IsTrue(rotations.All(CharacterMotionSpace.IsFinite));
                Assert.IsTrue(diagnostics.Diagnostics.Any(value => value.Code == "optional_bone_fallback"));

                void AddArm(bool left)
                {
                    float sign = left ? -1f : 1f;
                    Add(left ? HumanBodyBones.LeftShoulder : HumanBodyBones.RightShoulder, new Vector3(sign * 0.15f, 1.5f, 0f));
                    Add(left ? HumanBodyBones.LeftUpperArm : HumanBodyBones.RightUpperArm, new Vector3(sign * 0.35f, 1.5f, 0f));
                    Add(left ? HumanBodyBones.LeftLowerArm : HumanBodyBones.RightLowerArm, new Vector3(sign * 0.6f, 1.5f, 0f));
                    Add(left ? HumanBodyBones.LeftHand : HumanBodyBones.RightHand, new Vector3(sign * 0.8f, 1.5f, 0f));
                }
                void AddLeg(bool left)
                {
                    float sign = left ? -1f : 1f;
                    Add(left ? HumanBodyBones.LeftUpperLeg : HumanBodyBones.RightUpperLeg, new Vector3(sign * 0.1f, 0.9f, 0f));
                    Add(left ? HumanBodyBones.LeftLowerLeg : HumanBodyBones.RightLowerLeg, new Vector3(sign * 0.1f, 0.5f, 0f));
                    Add(left ? HumanBodyBones.LeftFoot : HumanBodyBones.RightFoot, new Vector3(sign * 0.1f, 0.05f, 0.1f));
                }
            }
            finally { foreach (GameObject value in created) UnityEngine.Object.DestroyImmediate(value); }
        }

        [Test]
        public void CharacterMotion_KeyframesAreSortedByFrameNotHierarchy()
        {
            var root = new GameObject("motion");
            try
            {
                CharacterMotion motion = root.AddComponent<CharacterMotion>();
                Add(40); Add(3); Add(25);
                CollectionAssert.AreEqual(new[] { 3, 25, 40 }, motion.GetKeyframes().Select(value => value.Frame).ToArray());
                void Add(int frame)
                {
                    var child = new GameObject("frame"); child.transform.SetParent(root.transform);
                    child.AddComponent<CharacterMotionKeyframe>().Configure(frame, CharacterMotionConstraintType.RootPosition);
                }
            }
            finally { UnityEngine.Object.DestroyImmediate(root); }
        }

        [Test]
        [Category("CharacterMotionProjectIntegration")]
        public void CharacterMotion_CompilesCapturedPoseWithoutEditorSkeleton()
        {
            const string prefabPath = "Assets/Bundles/Nyx/Prefabs/Nyx.prefab";
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
            if (prefab == null) Assert.Ignore($"Project Humanoid fixture not found at {prefabPath}.");
            GameObject character = null;
            CharacterMotionIntent intent = null;
            try
            {
                character = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                Animator animator = character.GetComponentInChildren<Animator>(true);
                Assert.NotNull(animator);
                Assert.IsTrue(animator.avatar != null && animator.avatar.isHuman);

                var motionObject = new GameObject("AuthoringMotion");
                motionObject.transform.SetParent(character.transform.parent, false);
                CharacterMotion motion = motionObject.AddComponent<CharacterMotion>();
                intent = ScriptableObject.CreateInstance<CharacterMotionIntent>();
                SetPrompt(intent, "A person holds a constrained pose.");
                var motionSerialized = new SerializedObject(motion);
                motionSerialized.FindProperty("intent").objectReferenceValue = intent;
                motionSerialized.FindProperty("targetAnimator").objectReferenceValue = animator;
                motionSerialized.FindProperty("actionOrigin").objectReferenceValue = motionObject.transform;
                motionSerialized.ApplyModifiedPropertiesWithoutUndo();

                var frameObject = new GameObject("Frame_12");
                frameObject.transform.SetParent(motionObject.transform, false);
                Transform hips = animator.GetBoneTransform(HumanBodyBones.Hips);
                frameObject.transform.SetPositionAndRotation(hips.position, animator.transform.rotation);
                CharacterMotionKeyframe frame = frameObject.AddComponent<CharacterMotionKeyframe>();
                frame.Configure(12, CharacterMotionConstraintType.RootPosition |
                    CharacterMotionConstraintType.FullBodyPose | CharacterMotionConstraintType.LeftHand);

                int count = (int)HumanBodyBones.LastBone;
                var humanoidRotations = new Quaternion[count];
                var humanoidAvailability = new bool[count];
                for (int i = 0; i < count; i++)
                {
                    Transform bone = animator.GetBoneTransform((HumanBodyBones)i);
                    humanoidRotations[i] = bone != null ? bone.localRotation : Quaternion.identity;
                    humanoidAvailability[i] = bone != null;
                }
                var diagnostics = new CharacterMotionValidationResult();
                Assert.IsTrue(HumanoidSomaMapper.TryCapture(animator.GetBoneTransform,
                    out Vector3[] somaPositions, out Quaternion[] somaRotations, diagnostics), string.Join("\n", diagnostics.Diagnostics));
                Quaternion inverse = Quaternion.Inverse(frameObject.transform.rotation);
                for (int i = 0; i < 30; i++)
                {
                    somaPositions[i] = inverse * (somaPositions[i] - frameObject.transform.position);
                    somaRotations[i] = inverse * somaRotations[i];
                }
                frame.Pose.Set(humanoidRotations, humanoidAvailability, somaPositions, somaRotations, motion.AvatarSignature);

                var handHandle = new GameObject("LeftHand");
                handHandle.transform.SetParent(frameObject.transform, false);
                Transform hand = animator.GetBoneTransform(HumanBodyBones.LeftHand);
                handHandle.transform.SetPositionAndRotation(hand.position, hand.rotation);
                frame.SetEffector(CharacterMotionConstraintType.LeftHand, handHandle.transform);

                Assert.IsNull(motionObject.transform.Find("MotionSkeleton"));
                KimodoConstraintSet set = motion.BuildConstraintSet();
                Assert.AreEqual(3, set.Constraints.Count);
                KimodoConditioning conditioning = new KimodoConstraintCompiler().Compile(set);
                Assert.IsTrue(conditioning.HasConstraints);
                UnityEngine.Object.DestroyImmediate(motionObject);
            }
            finally
            {
                if (character != null) UnityEngine.Object.DestroyImmediate(character);
                if (intent != null) UnityEngine.Object.DestroyImmediate(intent);
            }
        }

        [Test]
        [Category("CharacterMotionProjectIntegration")]
        public void EditorSkeleton_CapturesPoseAndIsEditorOnly()
        {
            const string prefabPath = "Assets/Bundles/Nyx/Prefabs/Nyx.prefab";
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
            if (prefab == null) Assert.Ignore($"Project Humanoid fixture not found at {prefabPath}.");
            GameObject character = null, motionObject = null;
            CharacterMotionIntent intent = null;
            try
            {
                character = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                Animator animator = character.GetComponentInChildren<Animator>(true);
                motionObject = new GameObject("EditorAuthoringMotion");
                CharacterMotion motion = motionObject.AddComponent<CharacterMotion>();
                intent = ScriptableObject.CreateInstance<CharacterMotionIntent>();
                AssignMotion(motion, intent, animator, motionObject.transform);

                CharacterMotionSkeleton skeleton = CharacterMotionSkeletonUtility.Rebuild(motion);
                Assert.AreEqual("EditorOnly", skeleton.gameObject.tag);
                Assert.IsTrue((skeleton.gameObject.hideFlags & HideFlags.DontSaveInBuild) != 0);
                Assert.NotNull(skeleton.GuideAnimator.GetBoneTransform(HumanBodyBones.Hips));

                var frameObject = new GameObject("Frame_00");
                frameObject.transform.SetParent(motionObject.transform, false);
                Transform hips = skeleton.GuideAnimator.GetBoneTransform(HumanBodyBones.Hips);
                frameObject.transform.SetPositionAndRotation(hips.position, skeleton.transform.rotation);
                CharacterMotionKeyframe frame = frameObject.AddComponent<CharacterMotionKeyframe>();
                frame.Configure(0, CharacterMotionConstraintType.FullBodyPose);
                skeleton.CapturePose(frame);
                Assert.IsTrue(frame.Pose.IsCaptured);
                Assert.IsTrue(frame.Pose.MatchesAvatar(motion.AvatarSignature));

                Transform hand = skeleton.GuideAnimator.GetBoneTransform(HumanBodyBones.LeftHand);
                hand.localRotation *= Quaternion.Euler(0f, 15f, 0f);
                Assert.IsTrue(skeleton.HasUnsavedChanges());
                skeleton.LoadPose(frame);
                Assert.IsFalse(skeleton.HasUnsavedChanges());
            }
            finally
            {
                if (motionObject != null) UnityEngine.Object.DestroyImmediate(motionObject);
                if (character != null) UnityEngine.Object.DestroyImmediate(character);
                if (intent != null) UnityEngine.Object.DestroyImmediate(intent);
            }
        }

        [Test]
        [Category("CharacterMotionProjectIntegration")]
        public void MotionPreview_SamplesOnlyGuideAndKeepsAvatarSignatureStable()
        {
            const string prefabPath = "Assets/Bundles/Nyx/Prefabs/Nyx.prefab";
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
            if (prefab == null) Assert.Ignore($"Project Humanoid fixture not found at {prefabPath}.");
            GameObject character = null, motionObject = null;
            CharacterMotionIntent intent = null;
            AnimationClip clip = null;
            string clipPath = null;
            try
            {
                character = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                Animator animator = character.GetComponentInChildren<Animator>(true);
                motionObject = new GameObject("PreviewMotion");
                CharacterMotion motion = motionObject.AddComponent<CharacterMotion>();
                intent = ScriptableObject.CreateInstance<CharacterMotionIntent>();
                AssignMotion(motion, intent, animator, motionObject.transform);
                CharacterMotionSkeleton skeleton = CharacterMotionSkeletonUtility.Rebuild(motion);
                Transform sourceHand = animator.GetBoneTransform(HumanBodyBones.RightHand);
                Transform guideHand = skeleton.GuideAnimator.GetBoneTransform(HumanBodyBones.RightHand);
                Assert.NotNull(sourceHand);
                Assert.NotNull(guideHand);
                Quaternion sourceRotation = sourceHand.localRotation;
                Quaternion guideRotation = guideHand.localRotation;
                Quaternion targetRotation = guideRotation * Quaternion.Euler(0f, 30f, 0f);
                string signature = motion.AvatarSignature;

                clip = new AnimationClip { frameRate = 30f };
                clipPath = AssetDatabase.GenerateUniqueAssetPath("Assets/CharacterMotionPreviewTest.anim");
                AssetDatabase.CreateAsset(clip, clipPath);
                string path = RelativePath(animator.transform, sourceHand);
                SetRotationCurve("x", guideRotation.x, targetRotation.x);
                SetRotationCurve("y", guideRotation.y, targetRotation.y);
                SetRotationCurve("z", guideRotation.z, targetRotation.z);
                SetRotationCurve("w", guideRotation.w, targetRotation.w);
                motion.SetBakedClip(clip, "test", "test");

                CharacterMotionEditor.Preview(motion, 1f);

                Assert.Less(Quaternion.Angle(sourceRotation, sourceHand.localRotation), 1e-4f,
                    "Preview must not sample transform curves onto the source armature.");
                Assert.Less(Quaternion.Angle(targetRotation, guideHand.localRotation), 0.1f,
                    "Preview should sample the baked clip onto the editor-only guide.");
                Assert.AreEqual(signature, motion.AvatarSignature,
                    "Avatar identity must not depend on mutable preview transforms.");

                void SetRotationCurve(string component, float from, float to)
                {
                    var binding = EditorCurveBinding.FloatCurve(path, typeof(Transform), "m_LocalRotation." + component);
                    AnimationUtility.SetEditorCurve(clip, binding, AnimationCurve.Linear(0f, from, 1f, to));
                }
            }
            finally
            {
                CharacterMotionEditor.StopPreview();
                if (!string.IsNullOrEmpty(clipPath)) AssetDatabase.DeleteAsset(clipPath);
                else if (clip != null) UnityEngine.Object.DestroyImmediate(clip);
                if (motionObject != null) UnityEngine.Object.DestroyImmediate(motionObject);
                if (character != null) UnityEngine.Object.DestroyImmediate(character);
                if (intent != null) UnityEngine.Object.DestroyImmediate(intent);
            }
        }

        [Test]
        public void ClipBaker_HumanoidClipUsesMusclesAndSurvivesRebake()
        {
            const string prefabPath = "Packages/ai.kitsumate.onnx.motion/Tests/PlayMode/Resources/XBot.fbx";
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
            if (prefab == null) Assert.Ignore($"Humanoid fixture not found at {prefabPath}.");
            GameObject character = null, motionObject = null;
            CharacterMotionIntent intent = null;
            string clipPath = null;
            try
            {
                character = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                Animator animator = character.GetComponentInChildren<Animator>(true);
                Assert.NotNull(animator);
                Assert.IsTrue(animator.avatar != null && animator.avatar.isHuman);

                motionObject = new GameObject("HumanoidClipBakeMotion");
                CharacterMotion motion = motionObject.AddComponent<CharacterMotion>();
                intent = ScriptableObject.CreateInstance<CharacterMotionIntent>();
                AssignMotion(motion, intent, animator, motionObject.transform);
                CharacterMotionSkeleton skeleton = CharacterMotionSkeletonUtility.Rebuild(motion);
                using var bindPoseHandler = new HumanPoseHandler(animator.avatar, animator.transform);
                var bindPose = new HumanPose();
                bindPoseHandler.GetHumanPose(ref bindPose);
                float bindBodyZ = bindPose.bodyPosition.z;
                float bindLowestFoot = LowestFootOrToe(animator);

                int frames = 3, bones = (int)HumanBodyBones.LastBone;
                var rotations = Enumerable.Repeat(Quaternion.identity, frames * bones).ToArray();
                rotations[bones + (int)HumanBodyBones.RightUpperArm] = Quaternion.Euler(0f, 0f, 30f);
                rotations[2 * bones + (int)HumanBodyBones.RightUpperArm] = Quaternion.Euler(0f, 0f, 60f);
                var available = new bool[bones];
                available[(int)HumanBodyBones.RightUpperArm] = true;
                float hipsHeight = animator.transform.InverseTransformPoint(
                    animator.GetBoneTransform(HumanBodyBones.Hips).position).y;
                var roots = new[]
                {
                    new Vector3(0.2f, hipsHeight - 0.25f, -0.4f),
                    new Vector3(0.2f, hipsHeight - 0.25f, 0.1f),
                    new Vector3(0.2f, hipsHeight - 0.25f, 0.6f),
                };
                var smoothedRoots = new[] { Vector3.zero, Vector3.forward * 0.5f, Vector3.forward };
                var rootRotations = Enumerable.Repeat(Quaternion.identity, frames).ToArray();
                var generated = new KimodoHumanoidMotion(frames, 30f, rotations, roots, rootRotations, available,
                    smoothedRootPositions: smoothedRoots);

                Quaternion storedGuideRotation = skeleton.GuideAnimator.GetBoneTransform(HumanBodyBones.RightUpperArm).localRotation;
                AnimationClip first = CharacterMotionClipBaker.WriteClip(motion, generated);
                motion.SetBakedClip(first, "test", "test");
                clipPath = AssetDatabase.GetAssetPath(first);
                AnimationClip second = CharacterMotionClipBaker.WriteClip(motion, generated);

                Assert.AreSame(first, second);
                Assert.IsTrue(second.humanMotion);
                EditorCurveBinding[] bindings = AnimationUtility.GetCurveBindings(second);
                Assert.IsTrue(bindings.Any(binding => binding.type == typeof(Animator) && binding.propertyName == "RootT.z"));
                Assert.IsTrue(bindings.Any(binding => binding.type == typeof(Animator) &&
                    HumanTrait.MuscleName.Contains(binding.propertyName)));
                Assert.IsFalse(bindings.Any(binding => binding.type == typeof(Transform)));
                Assert.Less(Quaternion.Angle(storedGuideRotation,
                    skeleton.GuideAnimator.GetBoneTransform(HumanBodyBones.RightUpperArm).localRotation), 1e-4f,
                    "Baking must restore the guide pose even when rebaking an existing clip.");

                CharacterMotionEditor.Preview(motion, 2f / 30f);
                Assert.Greater(Quaternion.Angle(storedGuideRotation,
                    skeleton.GuideAnimator.GetBoneTransform(HumanBodyBones.RightUpperArm).localRotation), 1f,
                    "Humanoid preview should evaluate muscle curves on the guide Avatar.");
                using var sampledPoseHandler = new HumanPoseHandler(skeleton.GuideAnimator.avatar, skeleton.GuideAnimator.transform);
                var sampledPose = new HumanPose();
                sampledPoseHandler.GetHumanPose(ref sampledPose);
                Assert.AreEqual(bindBodyZ + 1f, sampledPose.bodyPosition.z, 0.06f,
                    "Humanoid locomotion must follow the smoothed trajectory instead of pelvis-local XZ displacement.");
                Assert.AreEqual(bindLowestFoot, LowestFootOrToe(skeleton.GuideAnimator), 1e-3f,
                    "Baking should preserve the target avatar's bind-pose floor clearance.");

                static float LowestFootOrToe(Animator value)
                {
                    float lowest = float.PositiveInfinity;
                    foreach (HumanBodyBones bone in new[] { HumanBodyBones.LeftFoot, HumanBodyBones.RightFoot,
                                 HumanBodyBones.LeftToes, HumanBodyBones.RightToes })
                    {
                        Transform transform = value.GetBoneTransform(bone);
                        if (transform != null) lowest = Mathf.Min(lowest, transform.position.y);
                    }
                    return lowest;
                }
            }
            finally
            {
                CharacterMotionEditor.StopPreview();
                if (!string.IsNullOrEmpty(clipPath)) AssetDatabase.DeleteAsset(clipPath);
                if (motionObject != null) UnityEngine.Object.DestroyImmediate(motionObject);
                if (character != null) UnityEngine.Object.DestroyImmediate(character);
                if (intent != null) UnityEngine.Object.DestroyImmediate(intent);
            }
        }

        [Test]
        public void ClipBaker_RebakePreservesAssetReferenceAndWritesRootCurves()
        {
            var root = new GameObject("ClipBakeMotion");
            CharacterMotionIntent intent = null;
            string path = null;
            bool generatedRootExisted = AssetDatabase.IsValidFolder("Assets/Generated");
            bool generatedFolderExisted = AssetDatabase.IsValidFolder("Assets/Generated/CharacterMotion");
            try
            {
                Animator animator = root.AddComponent<Animator>();
                CharacterMotion motion = root.AddComponent<CharacterMotion>();
                SetClipFormat(motion, CharacterMotionClipFormat.AvatarTransforms);
                intent = ScriptableObject.CreateInstance<CharacterMotionIntent>();
                AssignMotion(motion, intent, animator, root.transform);
                int frames = 3, bones = (int)HumanBodyBones.LastBone;
                var rotations = Enumerable.Repeat(Quaternion.identity, frames * bones).ToArray();
                var roots = new[] { Vector3.zero, Vector3.forward, Vector3.forward * 2f };
                var rootRotations = new[] { Quaternion.identity, Quaternion.Euler(0f, 30f, 0f), Quaternion.Euler(0f, 60f, 0f) };
                var generated = new KimodoHumanoidMotion(frames, 30f, rotations, roots, rootRotations, new bool[bones]);

                AnimationClip first = CharacterMotionClipBaker.WriteClip(motion, generated);
                motion.SetBakedClip(first, "test", "test");
                path = AssetDatabase.GetAssetPath(first);
                AnimationClip second = CharacterMotionClipBaker.WriteClip(motion, generated);
                Assert.AreSame(first, second);
                Assert.AreEqual(path, AssetDatabase.GetAssetPath(second));
                Assert.IsTrue(AnimationUtility.GetCurveBindings(second).Any(binding => binding.propertyName == "m_LocalPosition.z"));
                Assert.AreEqual(30f, second.frameRate);
                Assert.AreEqual(0.1f, second.length, 1e-5f, "The last generated sample must be held for one complete frame.");
            }
            finally
            {
                if (!string.IsNullOrEmpty(path)) AssetDatabase.DeleteAsset(path);
                if (!generatedFolderExisted) AssetDatabase.DeleteAsset("Assets/Generated/CharacterMotion");
                if (!generatedRootExisted) AssetDatabase.DeleteAsset("Assets/Generated");
                UnityEngine.Object.DestroyImmediate(root);
                if (intent != null) UnityEngine.Object.DestroyImmediate(intent);
            }
        }

        [Test]
        [Category("CharacterMotionProjectIntegration")]
        public void ClipBaker_ConvertsGeneratedHipsIntoGroundedAnimatorRoot()
        {
            const string prefabPath = "Assets/Bundles/Nyx/Prefabs/Nyx.prefab";
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
            if (prefab == null) Assert.Ignore($"Project Humanoid fixture not found at {prefabPath}.");
            GameObject character = null, motionObject = null;
            CharacterMotionIntent intent = null;
            string clipPath = null;
            bool generatedRootExisted = AssetDatabase.IsValidFolder("Assets/Generated");
            bool generatedFolderExisted = AssetDatabase.IsValidFolder("Assets/Generated/CharacterMotion");
            try
            {
                character = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                Animator animator = character.GetComponentInChildren<Animator>(true);
                motionObject = new GameObject("GroundedBakeMotion");
                motionObject.transform.SetPositionAndRotation(new Vector3(3f, 0f, 4f), Quaternion.Euler(0f, 90f, 0f));
                CharacterMotion motion = motionObject.AddComponent<CharacterMotion>();
                SetClipFormat(motion, CharacterMotionClipFormat.AvatarTransforms);
                intent = ScriptableObject.CreateInstance<CharacterMotionIntent>();
                AssignMotion(motion, intent, animator, motionObject.transform);

                int bones = (int)HumanBodyBones.LastBone;
                var rotations = Enumerable.Repeat(Quaternion.identity, 2 * bones).ToArray();
                var generatedHips = new[] { new Vector3(0f, 1.1f, 0f), new Vector3(0f, 1.1f, 2f) };
                var generatedRotations = new[] { Quaternion.identity, Quaternion.Euler(0f, 15f, 0f) };
                var generated = new KimodoHumanoidMotion(2, 30f, rotations, generatedHips, generatedRotations, new bool[bones]);

                AnimationClip clip = CharacterMotionClipBaker.WriteClip(motion, generated);
                clipPath = AssetDatabase.GetAssetPath(clip);
                Vector3 bindHipsOffset = animator.transform.InverseTransformPoint(
                    animator.GetBoneTransform(HumanBodyBones.Hips).position);
                float scale = (Mathf.Abs(animator.transform.lossyScale.x) + Mathf.Abs(animator.transform.lossyScale.y) +
                               Mathf.Abs(animator.transform.lossyScale.z)) / 3f;

                for (int frame = 0; frame < 2; frame++)
                {
                    float time = frame / 30f;
                    Vector3 localPosition = ReadVectorCurve(clip, time);
                    Quaternion localRotation = ReadQuaternionCurve(clip, time);
                    Transform parent = animator.transform.parent;
                    Vector3 worldPosition = parent != null ? parent.TransformPoint(localPosition) : localPosition;
                    Quaternion worldRotation = parent != null ? parent.rotation * localRotation : localRotation;
                    Vector3 reconstructedHips = worldPosition + worldRotation * (bindHipsOffset * scale);
                    Vector3 expectedHips = motion.ActionOrigin.position + motion.ActionOrigin.rotation * generatedHips[frame];
                    AssertVector(expectedHips, reconstructedHips);
                    Assert.Less(Quaternion.Angle(motion.ActionOrigin.rotation * generatedRotations[frame], worldRotation), 1e-3f);
                }
            }
            finally
            {
                if (!string.IsNullOrEmpty(clipPath)) AssetDatabase.DeleteAsset(clipPath);
                if (!generatedFolderExisted) AssetDatabase.DeleteAsset("Assets/Generated/CharacterMotion");
                if (!generatedRootExisted) AssetDatabase.DeleteAsset("Assets/Generated");
                if (motionObject != null) UnityEngine.Object.DestroyImmediate(motionObject);
                if (character != null) UnityEngine.Object.DestroyImmediate(character);
                if (intent != null) UnityEngine.Object.DestroyImmediate(intent);
            }
        }

        private static void SetPrompt(CharacterMotionIntent intent, string value)
        {
            var serialized = new SerializedObject(intent);
            serialized.FindProperty("prompt").stringValue = value;
            serialized.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void SetClipFormat(CharacterMotion motion, CharacterMotionClipFormat format)
        {
            var serialized = new SerializedObject(motion);
            serialized.FindProperty("clipFormat").enumValueIndex = (int)format;
            serialized.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void AssignMotion(CharacterMotion motion, CharacterMotionIntent intent, Animator animator, Transform origin)
        {
            var serialized = new SerializedObject(motion);
            serialized.FindProperty("intent").objectReferenceValue = intent;
            serialized.FindProperty("targetAnimator").objectReferenceValue = animator;
            serialized.FindProperty("actionOrigin").objectReferenceValue = origin;
            serialized.ApplyModifiedPropertiesWithoutUndo();
        }

        private static string RelativePath(Transform root, Transform child)
        {
            var parts = new Stack<string>();
            for (Transform current = child; current != null && current != root; current = current.parent)
                parts.Push(current.name);
            return string.Join("/", parts);
        }

        private static Vector3 ReadVectorCurve(AnimationClip clip, float time)
        {
            float Read(string axis) => AnimationUtility.GetEditorCurve(clip,
                EditorCurveBinding.FloatCurve(string.Empty, typeof(Transform), "m_LocalPosition." + axis)).Evaluate(time);
            return new Vector3(Read("x"), Read("y"), Read("z"));
        }

        private static Quaternion ReadQuaternionCurve(AnimationClip clip, float time)
        {
            float Read(string axis) => AnimationUtility.GetEditorCurve(clip,
                EditorCurveBinding.FloatCurve(string.Empty, typeof(Transform), "m_LocalRotation." + axis)).Evaluate(time);
            return new Quaternion(Read("x"), Read("y"), Read("z"), Read("w")).normalized;
        }

        private static void AssertVector(Vector3 expected, Vector3 actual)
        {
            Assert.AreEqual(expected.x, actual.x, 1e-5f);
            Assert.AreEqual(expected.y, actual.y, 1e-5f);
            Assert.AreEqual(expected.z, actual.z, 1e-5f);
        }
    }
}
