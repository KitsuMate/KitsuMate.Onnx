using System;
using System.IO;
using KitsuMate.Onnx.Motion.Kimodo;
using KitsuMate.Onnx.Embeddings;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Editor
{
    internal static class CharacterMotionClipBaker
    {
        private const string GeneratedFolder = "Assets/Generated/CharacterMotion";
        internal static async Awaitable BakeEmbeddingAsync(CharacterMotionIntent intent)
        {
            if (intent == null) throw new ArgumentNullException(nameof(intent));
            if (string.IsNullOrWhiteSpace(intent.Prompt)) throw new InvalidOperationException("Enter a prompt before baking the embedding.");
            EmbeddingEngine engine = intent.EmbeddingEngine;
            if (engine == null) throw new InvalidOperationException("Assign an embedding engine to the intent.");
            if (engine.Backend == null) throw new InvalidOperationException($"Embedding engine '{engine.name}' has no backend assigned.");
            if (engine.ModelSet == null) throw new InvalidOperationException($"Embedding engine '{engine.name}' has no model set assigned.");
            using InferenceEngineRuntime<EmbeddingRequest, EmbeddingResult> runtime = await engine.CreateRuntimeAsync();
            EmbeddingResult result = await runtime.RunAsync(new EmbeddingRequest(intent.Prompt));
            if (result.Embedding == null || result.Embedding.Length != KimodoTextEmbedding.Dimension)
                throw new InvalidOperationException($"The selected embedding engine returned {result.Dimension} values; Kimodo requires {KimodoTextEmbedding.Dimension}.");
            var embedding = new KimodoTextEmbedding(result.Embedding, engine.ModelSet.Identity.ToString());
            Undo.RecordObject(intent, "Bake Character Motion Embedding");
            intent.SetEmbedding(engine.ModelSet.Identity, embedding);
            EditorUtility.SetDirty(intent);
            AssetDatabase.SaveAssets();
        }

        internal static async Awaitable BakeAnimationAsync(CharacterMotion motion)
        {
            CharacterMotionValidationResult validation = motion.ValidateMotion();
            if (!validation.IsValid) throw new InvalidOperationException(string.Join("\n", validation.Diagnostics));
            if (motion.MotionEngine == null) throw new InvalidOperationException("Assign a Character Motion engine.");
            ModelIdentity encoderIdentity = motion.RequiredEmbeddingModelIdentity;
            if (!motion.Intent.TryGetEmbedding(encoderIdentity, out KimodoTextEmbedding embedding))
            {
                throw new InvalidOperationException(
                    $"The selected intent has no current embedding compatible with '{encoderIdentity}'. " +
                    "Select the CharacterMotionIntent and bake its embedding before baking animation.");
            }
            CharacterMotionSkeleton skeleton = motion.GetComponentInChildren<CharacterMotionSkeleton>(true);
            if (skeleton == null || skeleton.BindLocalRotations.Length != (int)HumanBodyBones.LastBone)
                throw new InvalidOperationException("Create or rebuild the Character Motion skeleton before baking so bind rotations are known.");
            if (!skeleton.HasCompleteBindPose)
                throw new InvalidOperationException("Rebuild the Character Motion skeleton before baking so its complete bind transform state is captured.");

            if (motion.MotionEngine.Backend == null)
                throw new InvalidOperationException($"Motion engine '{motion.MotionEngine.name}' has no backend assigned.");
            using InferenceEngineRuntime<CharacterMotionRequest, CharacterMotionResult> runtime = await motion.MotionEngine.CreateRuntimeAsync();
            CharacterMotionResult generated = await runtime.RunAsync(motion.BuildEngineRequest(embedding));
            AnimationClip clip = WriteClip(motion, generated.Motion);
            Undo.RecordObject(motion, "Assign Baked Character Motion");
            motion.SetBakedClip(clip, encoderIdentity.ToString(), generated.ModelIdentity.ToString());
            EditorUtility.SetDirty(motion);
            AssetDatabase.SaveAssets();
        }

        internal static AnimationClip WriteClip(CharacterMotion motion, KimodoHumanoidMotion generated)
        {
            EnsureFolder(GeneratedFolder);
            AnimationClip clip = motion.BakedClip;
            string existingPath = clip != null ? AssetDatabase.GetAssetPath(clip) : null;
            if (clip == null || string.IsNullOrEmpty(existingPath))
            {
                clip = new AnimationClip { name = motion.name + "_Generated", frameRate = generated.FramesPerSecond, wrapMode = WrapMode.ClampForever };
                string path = AssetDatabase.GenerateUniqueAssetPath($"{GeneratedFolder}/{Sanitize(motion.name)}.anim");
                AssetDatabase.CreateAsset(clip, path);
            }
            else
            {
                foreach (EditorCurveBinding binding in AnimationUtility.GetCurveBindings(clip))
                    AnimationUtility.SetEditorCurve(clip, binding, null);
            }

            clip.frameRate = generated.FramesPerSecond;
            clip.wrapMode = WrapMode.ClampForever;
            Animator animator = motion.TargetAnimator;
            CharacterMotionSkeleton skeleton = motion.GetComponentInChildren<CharacterMotionSkeleton>(true);
            if (animator == null) throw new InvalidOperationException("A target Animator is required to bake animation.");
            if (motion.ClipFormat == CharacterMotionClipFormat.Humanoid &&
                (animator.avatar == null || !animator.avatar.isHuman))
                throw new InvalidOperationException("A valid Humanoid target Animator is required to bake a Humanoid clip.");

            skeleton?.StopPreview();
            TransformState[] guideState = skeleton != null ? CaptureState(skeleton.transform) : null;
            try
            {
                ApplyStoredBindPose(skeleton);
                BuildPoseCurves(motion, generated, skeleton, out Quaternion[][] boneRotations,
                    out Vector3[] rootPositions, out Quaternion[] rootRotations);
                if (motion.ClipFormat == CharacterMotionClipFormat.Humanoid)
                    WriteHumanoidCurves(clip, motion, skeleton, boneRotations, rootPositions, rootRotations, generated.FramesPerSecond);
                else
                    WriteTransformCurves(clip, animator, generated, boneRotations, rootPositions, rootRotations);
            }
            finally
            {
                if (guideState != null) RestoreState(guideState);
            }

            clip.EnsureQuaternionContinuity();
            EditorUtility.SetDirty(clip);
            return clip;
        }

        private static void BuildPoseCurves(CharacterMotion motion, KimodoHumanoidMotion generated,
            CharacterMotionSkeleton skeleton, out Quaternion[][] boneRotations,
            out Vector3[] rootPositions, out Quaternion[] rootRotations)
        {
            Animator animator = motion.TargetAnimator;
            ReadOnlySpan<Quaternion> bind = skeleton != null ? skeleton.BindLocalRotations.Span : ReadOnlySpan<Quaternion>.Empty;
            var guideBones = new System.Collections.Generic.Dictionary<Transform, HumanBodyBones>();
            if (skeleton != null && skeleton.GuideAnimator != null)
                for (int i = 0; i < (int)HumanBodyBones.LastBone; i++)
                {
                    Transform guideBone = skeleton.GuideAnimator.GetBoneTransform((HumanBodyBones)i);
                    if (guideBone != null) guideBones[guideBone] = (HumanBodyBones)i;
                }

            int bones = (int)HumanBodyBones.LastBone;
            boneRotations = new Quaternion[bones][];
            for (int boneIndex = 0; boneIndex < bones; boneIndex++)
            {
                if (!generated.BoneAvailability[boneIndex]) continue;
                Transform bone = animator.GetBoneTransform((HumanBodyBones)boneIndex);
                if (bone == null) continue;
                Quaternion baseRotation = bind.Length == bones ? bind[boneIndex] : bone.localRotation;
                var values = new Quaternion[generated.FrameCount];
                for (int frame = 0; frame < values.Length; frame++)
                {
                    HumanBodyBones humanoidBone = (HumanBodyBones)boneIndex;
                    Quaternion value;
                    Transform guideBone = skeleton != null && skeleton.GuideAnimator != null
                        ? skeleton.GuideAnimator.GetBoneTransform(humanoidBone)
                        : null;
                    if (generated.BoneGlobalRotationDeltas != null && guideBone != null)
                    {
                        Quaternion inverseGuideRoot = Quaternion.Inverse(skeleton.transform.rotation);
                        Quaternion bindBoneGlobal = inverseGuideRoot * guideBone.rotation;
                        Transform guideParent = guideBone.parent;
                        Quaternion bindParentGlobal = guideParent == null || guideParent == skeleton.transform
                            ? Quaternion.identity
                            : inverseGuideRoot * guideParent.rotation;
                        Quaternion parentDelta = Quaternion.identity;
                        for (Transform ancestor = guideParent; ancestor != null && ancestor != skeleton.transform; ancestor = ancestor.parent)
                            if (guideBones.TryGetValue(ancestor, out HumanBodyBones parentBone) &&
                                generated.BoneAvailability[(int)parentBone])
                            {
                                parentDelta = generated.GetBoneGlobalRotation(frame, parentBone);
                                break;
                            }
                        Quaternion boneDelta = generated.GetBoneGlobalRotation(frame, humanoidBone);
                        value = Quaternion.Normalize(Quaternion.Inverse(parentDelta * bindParentGlobal) *
                                                     (boneDelta * bindBoneGlobal));
                    }
                    else value = Quaternion.Normalize(baseRotation * generated.GetBoneRotation(frame, humanoidBone));
                    if (frame > 0 && Quaternion.Dot(values[frame - 1], value) < 0f)
                        value = new Quaternion(-value.x, -value.y, -value.z, -value.w);
                    values[frame] = value;
                }
                boneRotations[boneIndex] = values;
            }

            Transform hips = animator.avatar != null && animator.avatar.isHuman
                ? animator.GetBoneTransform(HumanBodyBones.Hips)
                : null;
            Vector3 bindHipsOffset = hips != null
                ? animator.transform.InverseTransformPoint(hips.position)
                : Vector3.zero;
            Vector3 characterScale = animator.transform.lossyScale;
            float uniformScale = (Mathf.Abs(characterScale.x) + Mathf.Abs(characterScale.y) + Mathf.Abs(characterScale.z)) / 3f;
            Transform actionOrigin = motion.ActionOrigin;
            Transform parent = animator.transform.parent;
            rootPositions = new Vector3[generated.FrameCount];
            rootRotations = new Quaternion[generated.FrameCount];
            for (int frame = 0; frame < generated.FrameCount; frame++)
            {
                Quaternion worldRotation = Quaternion.Normalize(actionOrigin.rotation * generated.RootRotations[frame]);
                Vector3 pelvis = generated.RootPositions[frame];
                Vector3 trajectory = generated.SmoothedRootPositions[frame];
                // Kimodo's root constraint addresses the smoothed ground-plane trajectory. The
                // non-smoothed pelvis XZ also contains pose-local displacement and must not become
                // an offset on the exported character root.
                Vector3 generatedHipsWorld = actionOrigin.position + actionOrigin.rotation *
                    new Vector3(trajectory.x, pelvis.y, trajectory.z);
                Vector3 worldPosition = generatedHipsWorld - worldRotation * (bindHipsOffset * uniformScale);
                rootPositions[frame] = parent != null ? parent.InverseTransformPoint(worldPosition) : worldPosition;
                rootRotations[frame] = parent != null
                    ? Quaternion.Normalize(Quaternion.Inverse(parent.rotation) * worldRotation)
                    : worldRotation;
                if (frame > 0 && Quaternion.Dot(rootRotations[frame - 1], rootRotations[frame]) < 0f)
                {
                    Quaternion q = rootRotations[frame];
                    rootRotations[frame] = new Quaternion(-q.x, -q.y, -q.z, -q.w);
                }
            }

            AlignTrajectoryToAvatarFloor(motion, skeleton, boneRotations, rootPositions, rootRotations);
        }

        private static void AlignTrajectoryToAvatarFloor(CharacterMotion motion, CharacterMotionSkeleton skeleton,
            Quaternion[][] boneRotations, Vector3[] rootPositions, Quaternion[] rootRotations)
        {
            if (skeleton == null || skeleton.GuideAnimator == null || rootPositions.Length == 0) return;

            Animator guide = skeleton.GuideAnimator;
            float bindFootHeight = LowestFootOrToeHeight(guide);
            if (!float.IsFinite(bindFootHeight)) return;
            float bindClearance = bindFootHeight - guide.transform.position.y;
            float desiredLowestHeight = motion.ActionOrigin.position.y + bindClearance;
            float generatedLowestHeight = float.PositiveInfinity;
            Transform targetParent = motion.TargetAnimator.transform.parent;

            for (int frame = 0; frame < rootPositions.Length; frame++)
            {
                Vector3 worldPosition = targetParent != null
                    ? targetParent.TransformPoint(rootPositions[frame])
                    : rootPositions[frame];
                Quaternion worldRotation = targetParent != null
                    ? targetParent.rotation * rootRotations[frame]
                    : rootRotations[frame];
                guide.transform.SetPositionAndRotation(worldPosition, worldRotation);
                for (int boneIndex = 0; boneIndex < boneRotations.Length; boneIndex++)
                {
                    Quaternion[] values = boneRotations[boneIndex];
                    Transform bone = guide.GetBoneTransform((HumanBodyBones)boneIndex);
                    if (values != null && bone != null) bone.localRotation = values[frame];
                }
                generatedLowestHeight = Mathf.Min(generatedLowestHeight, LowestFootOrToeHeight(guide));
            }

            if (!float.IsFinite(generatedLowestHeight)) return;
            Vector3 worldCorrection = Vector3.up * (desiredLowestHeight - generatedLowestHeight);
            Vector3 localCorrection = targetParent != null
                ? targetParent.InverseTransformVector(worldCorrection)
                : worldCorrection;
            for (int frame = 0; frame < rootPositions.Length; frame++) rootPositions[frame] += localCorrection;
        }

        private static float LowestFootOrToeHeight(Animator animator)
        {
            float result = float.PositiveInfinity;
            HumanBodyBones[] bones =
            {
                HumanBodyBones.LeftFoot,
                HumanBodyBones.RightFoot,
                HumanBodyBones.LeftToes,
                HumanBodyBones.RightToes,
            };
            for (int i = 0; i < bones.Length; i++)
            {
                Transform bone = animator.GetBoneTransform(bones[i]);
                if (bone != null) result = Mathf.Min(result, bone.position.y);
            }
            return result;
        }

        private static void WriteTransformCurves(AnimationClip clip, Animator animator, KimodoHumanoidMotion generated,
            Quaternion[][] boneRotations, Vector3[] rootPositions, Quaternion[] rootRotations)
        {
            for (int boneIndex = 0; boneIndex < boneRotations.Length; boneIndex++)
            {
                Quaternion[] values = boneRotations[boneIndex];
                if (values == null) continue;
                Transform bone = animator.GetBoneTransform((HumanBodyBones)boneIndex);
                if (bone != null)
                    WriteQuaternion(clip, CharacterMotion.RelativePath(animator.transform, bone), values, generated.FramesPerSecond);
            }
            WriteVector3(clip, string.Empty, "m_LocalPosition", rootPositions, generated.FramesPerSecond);
            WriteQuaternion(clip, string.Empty, rootRotations, generated.FramesPerSecond);
        }

        private static void WriteHumanoidCurves(AnimationClip clip, CharacterMotion motion, CharacterMotionSkeleton skeleton,
            Quaternion[][] boneRotations, Vector3[] rootPositions, Quaternion[] rootRotations, float fps)
        {
            if (skeleton == null || skeleton.GuideAnimator == null || skeleton.GuideAnimator.avatar == null ||
                !skeleton.GuideAnimator.avatar.isHuman)
                throw new InvalidOperationException("Create or rebuild the Character Motion skeleton before baking a Humanoid clip.");

            Animator guide = skeleton.GuideAnimator;
            int frames = rootPositions.Length;
            int muscles = HumanTrait.MuscleCount;
            var muscleValues = new float[muscles][];
            for (int muscle = 0; muscle < muscles; muscle++) muscleValues[muscle] = new float[frames];
            var bodyPositions = new Vector3[frames];
            var bodyRotations = new Quaternion[frames];
            Transform targetParent = motion.TargetAnimator.transform.parent;

            using var handler = new HumanPoseHandler(guide.avatar, guide.transform);
            var pose = new HumanPose();
            for (int frame = 0; frame < frames; frame++)
            {
                Vector3 worldPosition = targetParent != null
                    ? targetParent.TransformPoint(rootPositions[frame])
                    : rootPositions[frame];
                Quaternion worldRotation = targetParent != null
                    ? targetParent.rotation * rootRotations[frame]
                    : rootRotations[frame];
                guide.transform.SetPositionAndRotation(worldPosition, worldRotation);
                for (int boneIndex = 0; boneIndex < boneRotations.Length; boneIndex++)
                {
                    Quaternion[] values = boneRotations[boneIndex];
                    Transform bone = guide.GetBoneTransform((HumanBodyBones)boneIndex);
                    if (values != null && bone != null) bone.localRotation = values[frame];
                }
                handler.GetHumanPose(ref pose);
                bodyPositions[frame] = pose.bodyPosition;
                bodyRotations[frame] = Quaternion.Normalize(pose.bodyRotation);
                if (frame > 0 && Quaternion.Dot(bodyRotations[frame - 1], bodyRotations[frame]) < 0f)
                {
                    Quaternion q = bodyRotations[frame];
                    bodyRotations[frame] = new Quaternion(-q.x, -q.y, -q.z, -q.w);
                }
                for (int muscle = 0; muscle < muscles; muscle++) muscleValues[muscle][frame] = pose.muscles[muscle];
            }

            WriteAnimatorVector3(clip, "RootT", bodyPositions, fps);
            WriteAnimatorQuaternion(clip, "RootQ", bodyRotations, fps);
            for (int muscle = 0; muscle < muscles; muscle++)
                WriteAnimatorFloat(clip, HumanTrait.MuscleName[muscle], muscleValues[muscle], fps);
        }

        private static void WriteQuaternion(AnimationClip clip, string path, Quaternion[] values, float fps)
        {
            string[] names = { "m_LocalRotation.x", "m_LocalRotation.y", "m_LocalRotation.z", "m_LocalRotation.w" };
            for (int axis = 0; axis < 4; axis++)
            {
                var keys = new Keyframe[values.Length + 1];
                for (int i = 0; i <= values.Length; i++)
                {
                    Quaternion sample = values[Math.Min(i, values.Length - 1)];
                    float value = axis == 0 ? sample.x : axis == 1 ? sample.y : axis == 2 ? sample.z : sample.w;
                    keys[i] = new Keyframe(i / fps, value);
                }
                AnimationUtility.SetEditorCurve(clip, EditorCurveBinding.FloatCurve(path, typeof(Transform), names[axis]), new AnimationCurve(keys));
            }
        }

        private static void WriteVector3(AnimationClip clip, string path, string prefix, Vector3[] values, float fps)
        {
            string[] axes = { ".x", ".y", ".z" };
            for (int axis = 0; axis < 3; axis++)
            {
                var keys = new Keyframe[values.Length + 1];
                for (int i = 0; i <= values.Length; i++)
                {
                    Vector3 sample = values[Math.Min(i, values.Length - 1)];
                    keys[i] = new Keyframe(i / fps, axis == 0 ? sample.x : axis == 1 ? sample.y : sample.z);
                }
                AnimationUtility.SetEditorCurve(clip, EditorCurveBinding.FloatCurve(path, typeof(Transform), prefix + axes[axis]), new AnimationCurve(keys));
            }
        }

        private static void WriteAnimatorVector3(AnimationClip clip, string prefix, Vector3[] values, float fps)
        {
            WriteAnimatorFloat(clip, prefix + ".x", Array.ConvertAll(values, value => value.x), fps);
            WriteAnimatorFloat(clip, prefix + ".y", Array.ConvertAll(values, value => value.y), fps);
            WriteAnimatorFloat(clip, prefix + ".z", Array.ConvertAll(values, value => value.z), fps);
        }

        private static void WriteAnimatorQuaternion(AnimationClip clip, string prefix, Quaternion[] values, float fps)
        {
            WriteAnimatorFloat(clip, prefix + ".x", Array.ConvertAll(values, value => value.x), fps);
            WriteAnimatorFloat(clip, prefix + ".y", Array.ConvertAll(values, value => value.y), fps);
            WriteAnimatorFloat(clip, prefix + ".z", Array.ConvertAll(values, value => value.z), fps);
            WriteAnimatorFloat(clip, prefix + ".w", Array.ConvertAll(values, value => value.w), fps);
        }

        private static void WriteAnimatorFloat(AnimationClip clip, string property, float[] values, float fps)
        {
            var keys = new Keyframe[values.Length + 1];
            for (int i = 0; i <= values.Length; i++)
                keys[i] = new Keyframe(i / fps, values[Math.Min(i, values.Length - 1)]);
            AnimationUtility.SetEditorCurve(clip,
                EditorCurveBinding.FloatCurve(string.Empty, typeof(Animator), property), new AnimationCurve(keys));
        }

        private static void ApplyStoredBindPose(CharacterMotionSkeleton skeleton)
        {
            skeleton?.ResetToBindPose();
        }

        private static TransformState[] CaptureState(Transform root)
        {
            Transform[] transforms = root.GetComponentsInChildren<Transform>(true);
            var result = new TransformState[transforms.Length];
            for (int i = 0; i < transforms.Length; i++) result[i] = new TransformState(transforms[i]);
            return result;
        }

        private static void RestoreState(TransformState[] state)
        {
            for (int i = 0; i < state.Length; i++) state[i].Restore();
        }

        private readonly struct TransformState
        {
            private readonly Transform transform;
            private readonly Vector3 localPosition;
            private readonly Quaternion localRotation;
            private readonly Vector3 localScale;

            public TransformState(Transform transform)
            {
                this.transform = transform;
                localPosition = transform.localPosition;
                localRotation = transform.localRotation;
                localScale = transform.localScale;
            }

            public void Restore()
            {
                if (transform == null) return;
                transform.localPosition = localPosition;
                transform.localRotation = localRotation;
                transform.localScale = localScale;
            }
        }

        private static void EnsureFolder(string path)
        {
            string[] parts = path.Split('/');
            string current = parts[0];
            for (int i = 1; i < parts.Length; i++)
            {
                string next = current + "/" + parts[i];
                if (!AssetDatabase.IsValidFolder(next)) AssetDatabase.CreateFolder(current, parts[i]);
                current = next;
            }
        }

        private static string Sanitize(string value)
        {
            foreach (char invalid in Path.GetInvalidFileNameChars()) value = value.Replace(invalid, '_');
            return string.IsNullOrWhiteSpace(value) ? "CharacterMotion" : value;
        }
    }
}
