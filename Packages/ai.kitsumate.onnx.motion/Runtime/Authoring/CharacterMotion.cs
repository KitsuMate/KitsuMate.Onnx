using System;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;
using System.Text;
using KitsuMate.Onnx.Motion.Kimodo;
using UnityEngine;

namespace KitsuMate.Onnx.Motion
{
    public enum CharacterMotionClipFormat
    {
        Humanoid,
        AvatarTransforms,
    }

    [DisallowMultipleComponent]
    public class CharacterMotion : MonoBehaviour
    {
        [SerializeField] private List<CharacterMotionPart> parts = new List<CharacterMotionPart>();
        [SerializeField] private CharacterMotion previousMotion;
        [SerializeField, Range(1, 19)] private int entryOverlapFrames = 5;
        [SerializeField, Range(1, 19)] private int internalOverlapFrames = 5;
        [SerializeField] private Animator targetAnimator;
        [SerializeField] private Transform actionOrigin;
        [SerializeField] private bool overrideGenerationSettings;
        [SerializeField] private CharacterMotionGenerationSettings generationSettings;
        [SerializeField] private CharacterMotionClipFormat clipFormat = CharacterMotionClipFormat.Humanoid;
        [SerializeField] private AnimationClip bakedClip;
        [SerializeField, HideInInspector] private float[] bakedHistory;
        [Header("Inference")]
        [SerializeField] private CharacterMotionEngine motionEngine;

        [Header("Bake metadata")]
        [SerializeField, HideInInspector] private string bakedAvatarSignature;
        [SerializeField, HideInInspector] private string bakedIntentHash;
        [SerializeField, HideInInspector] private string bakedConstraintHash;
        [SerializeField, HideInInspector] private string bakedEncoderIdentity;
        [SerializeField, HideInInspector] private string bakedModelIdentity;

        public CharacterMotionIntent Intent => parts != null && parts.Count > 0 && parts[0] != null
            ? parts[0].Intent
            : null;
        public IReadOnlyList<CharacterMotionPart> Parts => parts ?? (IReadOnlyList<CharacterMotionPart>)Array.Empty<CharacterMotionPart>();
        public CharacterMotion PreviousMotion => previousMotion;
        public int EntryOverlapFrames => entryOverlapFrames;
        public int InternalOverlapFrames => internalOverlapFrames;
        public Animator TargetAnimator => targetAnimator;
        public Transform ActionOrigin => actionOrigin != null ? actionOrigin : transform;
        public CharacterMotionClipFormat ClipFormat => clipFormat;
        public AnimationClip BakedClip => bakedClip;
        public CharacterMotionEngine MotionEngine => motionEngine;
        public ModelIdentity RequiredEmbeddingModelIdentity => motionEngine != null
            ? motionEngine.RequiredEmbeddingModelIdentity
            : default;
        public string AvatarSignature => ComputeAvatarSignature(targetAnimator);
        public bool IsBakedClipCurrent
        {
            get
            {
                var visited = new HashSet<CharacterMotion>();
                try
                {
                    for (CharacterMotion current = this; current != null; current = current.previousMotion)
                        if (!visited.Add(current) || !current.IsOwnBakeCurrent) return false;
                    return true;
                }
                catch (Exception error) when (error is ArgumentException || error is OverflowException)
                {
                    return false;
                }
            }
        }
        private bool IsOwnBakeCurrent => bakedClip != null &&
            string.Equals(bakedAvatarSignature, AvatarSignature, StringComparison.Ordinal) &&
            string.Equals(bakedIntentHash, ComputeIntentHash(), StringComparison.Ordinal) &&
            string.Equals(bakedConstraintHash, ComputeConstraintHash(), StringComparison.Ordinal) &&
            (motionEngine == null || motionEngine.ModelSet != null &&
                string.Equals(bakedModelIdentity, motionEngine.ModelSet.Identity.ToString(), StringComparison.Ordinal) &&
                string.Equals(bakedEncoderIdentity, RequiredEmbeddingModelIdentity.ToString(), StringComparison.Ordinal));
        public string BakedDependencyHash => Hash128.Compute(
            (bakedAvatarSignature ?? string.Empty) + "|" +
            (bakedIntentHash ?? string.Empty) + "|" +
            (bakedConstraintHash ?? string.Empty) + "|" +
            (bakedEncoderIdentity ?? string.Empty) + "|" +
            (bakedModelIdentity ?? string.Empty)).ToString();

        public CharacterMotionTimeline Timeline
        {
            get
            {
                CharacterMotionGenerationPlan plan = BuildGenerationPlan();
                return new CharacterMotionTimeline(plan.Runs.Length, plan.OutputFrameCount);
            }
        }

        /// <summary>Assigns the Humanoid that will receive generated motion and its action-space origin.</summary>
        public void ConfigureTarget(Animator animator, Transform origin = null)
        {
            targetAnimator = animator;
            actionOrigin = origin != null ? origin : transform;
        }

        /// <summary>Assigns the motion engine configuration, including its default backend.</summary>
        public void ConfigureInference(CharacterMotionEngine engine) => motionEngine = engine;

        /// <summary>Configures the preceding motion used to condition the entry history.</summary>
        public void ConfigurePrevious(CharacterMotion previous, int overlapFrames = 5)
        {
            CharacterMotionPlanner.ValidateOverlap(overlapFrames);
            previousMotion = previous;
            entryOverlapFrames = overlapFrames;
        }

        public void SetParts(IEnumerable<CharacterMotionPart> values)
        {
            parts = values != null ? new List<CharacterMotionPart>(values) : new List<CharacterMotionPart>();
        }

        public ICharacterMotionIntent ResolveIntent(ICharacterMotionIntent runtimeOverride = null)
        {
            ICharacterMotionIntent resolved = runtimeOverride ?? Intent;
            if (resolved == null) throw new InvalidOperationException("Character motion has no intent.");
            return resolved;
        }

        public CharacterMotionKeyframe[] GetKeyframes()
        {
            CharacterMotionKeyframe[] result = GetComponentsInChildren<CharacterMotionKeyframe>(true);
            Array.Sort(result, (a, b) => a.Frame.CompareTo(b.Frame));
            return result;
        }

        public CharacterMotionValidationResult ValidateMotion(bool requireIntent = true)
        {
            var result = new CharacterMotionValidationResult();
            CharacterMotionGenerationPlan plan;
            try
            {
                CharacterMotionPlanner.ValidateOverlap(entryOverlapFrames);
                CharacterMotionPlanner.ValidateOverlap(internalOverlapFrames);
                plan = BuildGenerationPlan();
            }
            catch (Exception error) when (error is ArgumentException || error is OverflowException)
            {
                result.Error("invalid_duration", error.Message);
                return result;
            }
            if (requireIntent && plan.Runs.Length == 0) result.Error("missing_intent", "Add at least one Character Motion intent.");
            for (int i = 0; i < plan.Runs.Length; i++)
                if (plan.Runs[i].Intent == null)
                    result.Error("missing_intent", $"Motion part {plan.Runs[i].PartIndex + 1} has no intent assigned.");
            ValidatePreviousMotion(result);
            if (targetAnimator == null) result.Error("missing_animator", "Assign a target Humanoid Animator.");
            else if (targetAnimator.avatar == null || !targetAnimator.avatar.isHuman)
                result.Error("non_humanoid_animator", "The target Animator must use a valid Humanoid Avatar.");

            ValidateScale(ActionOrigin.lossyScale, "action_origin_scale", result);
            if (targetAnimator != null) ValidateScale(targetAnimator.transform.lossyScale, "character_scale", result);

            CharacterMotionKeyframe[] frames = GetKeyframes();
            if (frames.Length == 0) result.Warning("no_keyframes", "The motion has no constraint keyframes.");
            var seen = new HashSet<int>();
            string avatar = AvatarSignature;
            for (int i = 0; i < frames.Length; i++)
            {
                CharacterMotionKeyframe keyframe = frames[i];
                if (plan.OutputFrameCount > 0 && keyframe.Frame >= plan.OutputFrameCount)
                    result.Error("frame_out_of_range", $"Frame {keyframe.Frame} is outside the {plan.OutputFrameCount}-frame motion timeline.");
                if (!seen.Add(keyframe.Frame)) result.Error("duplicate_frame", $"More than one keyframe uses frame {keyframe.Frame}.");
                if (keyframe.Constraints == CharacterMotionConstraintType.None)
                    result.Warning("empty_keyframe", $"Frame {keyframe.Frame} has no enabled constraints.");
                if (!CharacterMotionSpace.IsFinite(keyframe.transform.position) || !CharacterMotionSpace.IsFinite(keyframe.transform.rotation))
                    result.Error("non_finite_transform", $"Frame {keyframe.Frame} contains a non-finite transform.");
                Vector3 keyframeScale = keyframe.transform.lossyScale;
                if (!CharacterMotionSpace.IsFinite(keyframeScale) ||
                    Mathf.Max(Mathf.Abs(keyframeScale.x - 1f),
                        Mathf.Max(Mathf.Abs(keyframeScale.y - 1f), Mathf.Abs(keyframeScale.z - 1f))) > 1e-3f)
                    result.Error("keyframe_scale",
                        $"Frame {keyframe.Frame} must have unit world scale; its Transform supplies root position and rotation only.");
                if ((keyframe.Constraints & CharacterMotionConstraintType.RootHeading) != 0 &&
                    CharacterMotionSpace.WorldForwardToHeading(keyframe.transform.forward, ActionOrigin).sqrMagnitude < 0.5f)
                    result.Error("degenerate_heading", $"Frame {keyframe.Frame} has a degenerate root heading.");
                if (keyframe.RequiresPose && !keyframe.Pose.IsCaptured)
                    result.Error("missing_pose", $"Frame {keyframe.Frame} requires a captured pose.");
                else if (keyframe.RequiresPose && !keyframe.Pose.UsesCurrentRootSpace)
                    result.Error("stale_pose_space", $"Frame {keyframe.Frame} uses legacy guide-root pose data. Recalculate the motion.");
                else if (keyframe.RequiresPose && !keyframe.Pose.MatchesAvatar(avatar))
                    result.Error("stale_pose", $"Frame {keyframe.Frame} was captured for a different avatar.");
                ValidateEffector(keyframe, CharacterMotionConstraintType.LeftHand, result);
                ValidateEffector(keyframe, CharacterMotionConstraintType.RightHand, result);
                ValidateEffector(keyframe, CharacterMotionConstraintType.LeftFoot, result);
                ValidateEffector(keyframe, CharacterMotionConstraintType.RightFoot, result);
            }

            if (bakedClip != null && !IsBakedClipCurrent)
                result.Warning("stale_baked_clip", "The baked AnimationClip no longer matches the intent, avatar, or constraints.");
            return result;
        }

        public KimodoConstraintSet BuildConstraintSet()
        {
            CharacterMotionValidationResult validation = ValidateMotion(requireIntent: false);
            if (!validation.IsValid)
                throw new InvalidOperationException(string.Join("\n", validation.Diagnostics.Where(d => d.Severity == CharacterMotionDiagnosticSeverity.Error)));

            var constraints = new List<IKimodoConstraint>();
            CharacterMotionKeyframe[] frames = GetKeyframes();
            for (int i = 0; i < frames.Length; i++) AppendConstraints(frames[i], constraints);
            var set = new KimodoConstraintSet(constraints.ToArray());
            // Run the authoritative feature compiler now so overlaps/conflicts fail before inference.
            _ = new KimodoConstraintCompiler().Compile(set, Timeline.FrameCount == 0 ? 60 : Timeline.FrameCount);
            return set;
        }

        public CharacterMotionRequest BuildEngineRequest(KimodoTextEmbedding embedding = null, ICharacterMotionIntent runtimeIntent = null)
        {
            CharacterMotionGenerationPlan plan = BuildGenerationPlan();
            var segments = new List<CharacterMotionSegment>();
            if (runtimeIntent != null)
            {
                if (plan.Runs.Length > 1) throw new InvalidOperationException("A runtime intent override requires one motion part.");
                CharacterMotionGenerationSettings settings = overrideGenerationSettings
                    ? generationSettings.WithDefaults() : runtimeIntent.Settings;
                if (embedding == null && !runtimeIntent.TryGetEmbedding(RequiredEmbeddingModelIdentity, out embedding))
                    throw new InvalidOperationException("Runtime intent has no compatible embedding.");
                segments.Add(new CharacterMotionSegment(embedding, settings.CreateRequest(
                    plan.Runs.Length == 0 ? 60 : plan.Runs[0].FrameCount)));
            }
            else
            {
                foreach (var run in plan.Runs)
                {
                    if (run.Intent == null) throw new InvalidOperationException("Assign an intent to every motion part.");
                    KimodoTextEmbedding resolved = plan.Runs.Length == 1 ? embedding : null;
                    if (resolved == null && !run.Intent.TryGetEmbedding(RequiredEmbeddingModelIdentity, out resolved))
                        throw new InvalidOperationException($"Intent '{run.Intent.name}' has no compatible baked embedding.");
                    var settings = overrideGenerationSettings ? generationSettings.WithDefaults() : run.Settings;
                    segments.Add(new CharacterMotionSegment(resolved, settings.CreateRequest(run.FrameCount)));
                }
            }
            return new CharacterMotionRequest
            {
                Segments = segments.ToArray(), Constraints = BuildConstraintSet(),
                HistoryFrames = internalOverlapFrames, EntryHistoryFrames = entryOverlapFrames,
                PreviousMotion = PreviousSourceMotion()
            };
        }

        private KimodoHumanoidMotion PreviousSourceMotion()
        {
            if (previousMotion == null) return null;
            if (previousMotion.bakedHistory == null || previousMotion.bakedHistory.Length == 0)
                throw new InvalidOperationException("Rebake Previous Motion to provide canonical continuation data.");
            Vector3 position = CharacterMotionSpace.WorldToCanonical(previousMotion.ActionOrigin.position, ActionOrigin);
            Quaternion rotation = CharacterMotionSpace.WorldToCanonical(previousMotion.ActionOrigin.rotation, ActionOrigin);
            float[] history = KimodoMotionProcessing.TransformHistory(previousMotion.bakedHistory, position, rotation);
            var result = KimodoMotionDecoder.Decode(history, history.Length / 369, null);
            result.SetSourceMotion(history);
            return result;
        }

        internal void SetBakedHistory(KimodoHumanoidMotion motion)
        {
            if (motion?.SourceMotion == null) { bakedHistory = null; return; }
            int length = Math.Min(19, motion.FrameCount) * 369;
            bakedHistory = new float[length];
            Array.Copy(motion.SourceMotion, motion.SourceMotion.Length - length, bakedHistory, 0, length);
        }

        public string ComputeConstraintHash()
        {
            var builder = new StringBuilder();
            builder.Append(AvatarSignature).Append('|').Append((int)clipFormat).Append('|')
                .Append(entryOverlapFrames).Append('|').Append(internalOverlapFrames).Append('|')
                .Append(previousMotion != null ? previousMotion.BakedDependencyHash : string.Empty).Append('|');
            AppendTransform(builder, ActionOrigin);
            if (previousMotion != null) AppendTransform(builder, previousMotion.ActionOrigin);
            foreach (CharacterMotionKeyframe frame in GetKeyframes())
            {
                builder.Append(frame.Frame).Append(':').Append((int)frame.Constraints).Append(':');
                AppendVector(builder, frame.transform.position); builder.Append(':');
                AppendQuaternion(builder, frame.transform.rotation); builder.Append(':').Append(frame.Pose.ComputeHash());
                AppendTransform(builder, frame.LeftHand); AppendTransform(builder, frame.RightHand);
                AppendTransform(builder, frame.LeftFoot); AppendTransform(builder, frame.RightFoot);
            }
            return Hash128.Compute(builder.ToString()).ToString();
        }

        public string ComputeIntentHash()
        {
            var builder = new StringBuilder();
            CharacterMotionGenerationPlan plan = BuildGenerationPlan();
            for (int i = 0; i < plan.Runs.Length; i++)
            {
                CharacterMotionGenerationPart run = plan.Runs[i];
                CharacterMotionGenerationSettings s = overrideGenerationSettings
                    ? generationSettings.WithDefaults()
                    : run.Settings;
                builder.Append(run.PartIndex).Append(':').Append(run.RepetitionIndex).Append(':')
                    .Append(run.FrameCount).Append(':').Append(run.Intent != null ? run.Intent.Prompt : string.Empty).Append('|')
                    .Append(s.Seed).Append('|').Append(s.DenoisingSteps).Append('|')
                    .Append(s.TextGuidance.ToString("R", CultureInfo.InvariantCulture)).Append('|')
                    .Append(s.ConstraintGuidance.ToString("R", CultureInfo.InvariantCulture)).Append('|')
                    .Append(s.FirstHeadingRadians.ToString("R", CultureInfo.InvariantCulture)).Append(';');
            }
            return Hash128.Compute(builder.ToString()).ToString();
        }

        public void SetBakedClip(AnimationClip clip, string encoderIdentity, string modelIdentity)
        {
            bakedClip = clip;
            if (clip == null) bakedHistory = null;
            bakedAvatarSignature = AvatarSignature;
            bakedIntentHash = ComputeIntentHash();
            bakedConstraintHash = ComputeConstraintHash();
            bakedEncoderIdentity = encoderIdentity ?? string.Empty;
            bakedModelIdentity = modelIdentity ?? string.Empty;
        }

        public static string ComputeAvatarSignature(Animator animator)
        {
            if (animator == null || animator.avatar == null || !animator.avatar.isHuman) return string.Empty;
            HumanDescription description = animator.avatar.humanDescription;
            var humanToSkeleton = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (HumanBone bone in description.human)
                humanToSkeleton[bone.humanName] = bone.boneName;
            var bindSkeleton = new Dictionary<string, SkeletonBone>(StringComparer.Ordinal);
            foreach (SkeletonBone bone in description.skeleton)
                bindSkeleton[bone.name] = bone;

            var builder = new StringBuilder(animator.avatar.name);
            for (int i = 0; i < (int)HumanBodyBones.LastBone; i++)
            {
                Transform bone = animator.GetBoneTransform((HumanBodyBones)i);
                if (bone == null) { builder.Append('|').Append(i).Append("=missing"); continue; }
                if (!humanToSkeleton.TryGetValue(HumanTrait.BoneName[i], out string skeletonName) ||
                    !bindSkeleton.TryGetValue(skeletonName, out SkeletonBone bindBone))
                {
                    builder.Append('|').Append(i).Append("=missing-bind");
                    continue;
                }
                builder.Append('|').Append(i).Append('=').Append(RelativePath(animator.transform, bone))
                    .Append('@');
                AppendVector(builder, bindBone.position); builder.Append('@'); AppendVector(builder, bindBone.scale);
            }
            return Hash128.Compute(builder.ToString()).ToString();
        }

        private void AppendConstraints(CharacterMotionKeyframe keyframe, List<IKimodoConstraint> output)
        {
            int[] frame = { keyframe.Frame };
            Vector3 rootCanonical = CharacterMotionSpace.WorldToCanonical(keyframe.transform.position, ActionOrigin);
            Vector2 heading = CharacterMotionSpace.WorldForwardToHeading(keyframe.transform.forward, ActionOrigin);
            CharacterMotionConstraintType types = keyframe.Constraints;
            if ((types & (CharacterMotionConstraintType.RootPosition | CharacterMotionConstraintType.RootHeading)) != 0)
                output.Add(new KimodoRootConstraint(frame, new[] { new Vector2(rootCanonical.x, rootCanonical.z) },
                    (types & CharacterMotionConstraintType.RootHeading) != 0 ? new[] { heading } : null));

            if (!keyframe.RequiresPose) return;
            Vector3[] positions = ReconstructPositions(keyframe);
            Quaternion[] rotations = ReconstructRotations(keyframe);
            Vector2[] smoothRoot = { new Vector2(rootCanonical.x, rootCanonical.z) };
            if ((types & CharacterMotionConstraintType.FullBodyPose) != 0)
                output.Add(new KimodoFullBodyConstraint(frame, positions, smoothRoot));

            KimodoEndEffectors effectors = KimodoEndEffectors.None;
            ApplyEffector(keyframe, CharacterMotionConstraintType.LeftHand, KimodoJoint.LeftHand, KimodoJoint.LeftHandMiddleEnd, ref effectors, KimodoEndEffectors.LeftHand, positions, rotations);
            ApplyEffector(keyframe, CharacterMotionConstraintType.RightHand, KimodoJoint.RightHand, KimodoJoint.RightHandMiddleEnd, ref effectors, KimodoEndEffectors.RightHand, positions, rotations);
            ApplyEffector(keyframe, CharacterMotionConstraintType.LeftFoot, KimodoJoint.LeftFoot, KimodoJoint.LeftToeBase, ref effectors, KimodoEndEffectors.LeftFoot, positions, rotations);
            ApplyEffector(keyframe, CharacterMotionConstraintType.RightFoot, KimodoJoint.RightFoot, KimodoJoint.RightToeBase, ref effectors, KimodoEndEffectors.RightFoot, positions, rotations);
            if (effectors != KimodoEndEffectors.None)
                output.Add(new KimodoEndEffectorConstraint(frame, effectors, positions, rotations, smoothRoot));
        }

        private Vector3[] ReconstructPositions(CharacterMotionKeyframe keyframe)
        {
            ReadOnlySpan<Vector3> relative = keyframe.Pose.SomaPositionsRelativeToRoot.Span;
            var result = new Vector3[30];
            for (int i = 0; i < result.Length; i++)
                result[i] = CharacterMotionSpace.WorldToCanonical(keyframe.transform.position + keyframe.transform.rotation * relative[i], ActionOrigin);
            return result;
        }

        private Quaternion[] ReconstructRotations(CharacterMotionKeyframe keyframe)
        {
            ReadOnlySpan<Quaternion> relative = keyframe.Pose.SomaRotationsRelativeToRoot.Span;
            var result = new Quaternion[30];
            for (int i = 0; i < result.Length; i++)
                result[i] = CharacterMotionSpace.WorldToCanonical(keyframe.transform.rotation * relative[i], ActionOrigin);
            return result;
        }

        private void ApplyEffector(
            CharacterMotionKeyframe keyframe, CharacterMotionConstraintType type, KimodoJoint baseJoint, KimodoJoint endJoint,
            ref KimodoEndEffectors selected, KimodoEndEffectors flag, Vector3[] positions, Quaternion[] rotations)
        {
            if ((keyframe.Constraints & type) == 0) return;
            Transform handle = keyframe.GetEffector(type);
            int baseIndex = (int)baseJoint, endIndex = (int)endJoint;
            Vector3 oldBase = positions[baseIndex];
            Quaternion oldRotation = rotations[baseIndex];
            Vector3 newBase = CharacterMotionSpace.WorldToCanonical(handle.position, ActionOrigin);
            Quaternion newRotation = CharacterMotionSpace.WorldToCanonical(handle.rotation, ActionOrigin);
            Quaternion delta = newRotation * Quaternion.Inverse(oldRotation);
            positions[endIndex] = newBase + delta * (positions[endIndex] - oldBase);
            positions[baseIndex] = newBase;
            rotations[baseIndex] = newRotation;
            selected |= flag;
        }

        private static void ValidateEffector(CharacterMotionKeyframe frame, CharacterMotionConstraintType type, CharacterMotionValidationResult result)
        {
            if ((frame.Constraints & type) != 0 && frame.GetEffector(type) == null)
                result.Error("missing_effector", $"Frame {frame.Frame} enables {type} but has no handle.");
        }

        private static void ValidateScale(Vector3 scale, string code, CharacterMotionValidationResult result)
        {
            if (!CharacterMotionSpace.IsFinite(scale) || Mathf.Min(Mathf.Abs(scale.x), Mathf.Abs(scale.y), Mathf.Abs(scale.z)) < 1e-6f)
            { result.Error(code, "Scale must be finite and non-zero."); return; }
            if (scale.x < 0f || scale.y < 0f || scale.z < 0f)
            { result.Error(code, "Negative scale is not supported."); return; }
            if (Mathf.Max(Mathf.Abs(scale.x - scale.y), Mathf.Abs(scale.y - scale.z)) > 1e-3f)
                result.Error(code, "Non-uniform scale is not supported.");
            else if (Mathf.Abs(scale.x - 1f) > 0.01f)
                result.Warning(code, $"Uniform scale {scale.x:F3} differs from one; Unity world units are still treated as metres.");
        }

        private static void AppendTransform(StringBuilder builder, Transform value)
        {
            if (value == null) { builder.Append("|null"); return; }
            builder.Append('|'); AppendVector(builder, value.position); builder.Append('@'); AppendQuaternion(builder, value.rotation);
        }

        private static void AppendVector(StringBuilder builder, Vector3 value) => builder
            .Append(value.x.ToString("R", CultureInfo.InvariantCulture)).Append(',')
            .Append(value.y.ToString("R", CultureInfo.InvariantCulture)).Append(',')
            .Append(value.z.ToString("R", CultureInfo.InvariantCulture));

        private static void AppendQuaternion(StringBuilder builder, Quaternion value) => builder
            .Append(value.x.ToString("R", CultureInfo.InvariantCulture)).Append(',')
            .Append(value.y.ToString("R", CultureInfo.InvariantCulture)).Append(',')
            .Append(value.z.ToString("R", CultureInfo.InvariantCulture)).Append(',')
            .Append(value.w.ToString("R", CultureInfo.InvariantCulture));

        public static string RelativePath(Transform root, Transform target)
        {
            if (root == target) return string.Empty;
            var names = new Stack<string>();
            Transform current = target;
            while (current != null && current != root) { names.Push(current.name); current = current.parent; }
            return current == root ? string.Join("/", names) : target.name;
        }

        private void Reset()
        {
            targetAnimator = GetComponentInParent<Animator>();
            actionOrigin = transform;
            generationSettings = CharacterMotionGenerationSettings.Default;
        }

        internal CharacterMotionGenerationPlan BuildGenerationPlan()
        {
            return CharacterMotionPlanner.Build(parts);
        }

        private void ValidatePreviousMotion(CharacterMotionValidationResult result)
        {
            if (previousMotion == null) return;
            if (previousMotion == this)
            {
                result.Error("previous_motion_self_reference", "Previous Motion cannot reference this CharacterMotion.");
                return;
            }
            if (previousMotion.BakedClip == null || previousMotion.bakedHistory == null || previousMotion.bakedHistory.Length == 0)
                result.Error("missing_previous_motion_data", "Previous Motion has no baked animation data.");
            else if (!previousMotion.IsBakedClipCurrent)
                result.Error("stale_previous_motion", "Previous Motion must be rebaked before it can condition this motion.");
            if (targetAnimator != null && previousMotion.targetAnimator != null &&
                !string.Equals(AvatarSignature, previousMotion.AvatarSignature, StringComparison.Ordinal))
                result.Error("previous_motion_avatar_mismatch", "Previous Motion was authored for a different avatar.");
            var visited = new HashSet<CharacterMotion> { this };
            for (CharacterMotion current = previousMotion; current != null; current = current.previousMotion)
                if (!visited.Add(current))
                {
                    result.Error("previous_motion_cycle", "Previous Motion references form a cycle.");
                    break;
                }
        }

        private void OnValidate()
        {
            entryOverlapFrames = Mathf.Clamp(entryOverlapFrames, 1, 19);
            internalOverlapFrames = Mathf.Clamp(internalOverlapFrames, 1, 19);
        }
    }
}
