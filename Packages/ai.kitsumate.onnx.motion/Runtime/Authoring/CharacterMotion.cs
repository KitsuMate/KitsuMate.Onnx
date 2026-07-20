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
        [SerializeField, HideInInspector] private CharacterMotionIntent intent;
        [SerializeField] private List<CharacterMotionPart> parts = new List<CharacterMotionPart>();
        [SerializeField] private CharacterMotion previousMotion;
        [SerializeField, Range(0, KimodoConditioning.DefaultFrameCount - 1)] private int entryOverlapFrames = 5;
        [SerializeField, Range(0, KimodoConditioning.DefaultFrameCount - 1)] private int internalOverlapFrames = 5;
        [SerializeField] private Animator targetAnimator;
        [SerializeField] private Transform actionOrigin;
        [SerializeField] private bool overrideGenerationSettings;
        [SerializeField] private CharacterMotionGenerationSettings generationSettings;
        [SerializeField] private CharacterMotionClipFormat clipFormat = CharacterMotionClipFormat.Humanoid;
        [SerializeField] private AnimationClip bakedClip;
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
            : intent;
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
        public bool IsBakedClipCurrent => bakedClip != null &&
            string.Equals(bakedAvatarSignature, AvatarSignature, StringComparison.Ordinal) &&
            string.Equals(bakedIntentHash, ComputeIntentHash(), StringComparison.Ordinal) &&
            string.Equals(bakedConstraintHash, ComputeConstraintHash(), StringComparison.Ordinal);
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
                int effective = plan.OutputFrameCount - (previousMotion != null ? entryOverlapFrames : 0);
                return new CharacterMotionTimeline(plan.Runs.Length, plan.OutputFrameCount, Mathf.Max(0, effective));
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

        /// <summary>Configures the optional preceding motion used to condition and blend the entry overlap.</summary>
        public void ConfigurePrevious(CharacterMotion previous, int overlapFrames = 5)
        {
            if ((uint)overlapFrames >= KimodoConditioning.DefaultFrameCount)
                throw new ArgumentOutOfRangeException(nameof(overlapFrames));
            previousMotion = previous;
            entryOverlapFrames = overlapFrames;
        }

        public void SetParts(IEnumerable<CharacterMotionPart> values)
        {
            parts = values != null ? new List<CharacterMotionPart>(values) : new List<CharacterMotionPart>();
            intent = null;
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
            MigrateLegacyIntent();
            CharacterMotionGenerationPlan plan = BuildGenerationPlan();
            if (requireIntent && plan.Runs.Length == 0) result.Error("missing_intent", "Add at least one Character Motion intent.");
            for (int i = 0; i < plan.Runs.Length; i++)
                if (plan.Runs[i].Intent == null)
                    result.Error("missing_intent", $"Motion part {plan.Runs[i].PartIndex + 1} has no intent assigned.");
            ValidatePreviousMotion(result);
            if (plan.Runs.Length > 1)
                result.Error("multi_part_generation_pending",
                    "Multi-intent and repeated-intent authoring is configured, but multi-run generation is not implemented yet.");
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
                if ((keyframe.Constraints & CharacterMotionConstraintType.RootHeading) != 0 &&
                    CharacterMotionSpace.WorldForwardToHeading(keyframe.transform.forward, ActionOrigin).sqrMagnitude < 0.5f)
                    result.Error("degenerate_heading", $"Frame {keyframe.Frame} has a degenerate root heading.");
                if (keyframe.RequiresPose && !keyframe.Pose.IsCaptured)
                    result.Error("missing_pose", $"Frame {keyframe.Frame} requires a captured pose.");
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
            _ = new KimodoConstraintCompiler().Compile(set);
            return set;
        }

        public KimodoGenerationRequest BuildGenerationRequest(ICharacterMotionIntent runtimeIntent = null)
        {
            ICharacterMotionIntent resolved = ResolveIntent(runtimeIntent);
            CharacterMotionGenerationSettings settings = overrideGenerationSettings
                ? generationSettings.WithDefaults()
                : GetPrimarySettings(resolved);
            return settings.CreateRequest(BuildConstraintSet());
        }

        public CharacterMotionRequest BuildEngineRequest(KimodoTextEmbedding embedding, ICharacterMotionIntent runtimeIntent = null) => new CharacterMotionRequest
        {
            Embedding = embedding,
            Constraints = BuildConstraintSet(),
            Generation = BuildGenerationRequest(runtimeIntent)
        };

        public string ComputeConstraintHash()
        {
            var builder = new StringBuilder();
            builder.Append(AvatarSignature).Append('|').Append((int)clipFormat).Append('|')
                .Append(entryOverlapFrames).Append('|').Append(internalOverlapFrames).Append('|')
                .Append(previousMotion != null ? previousMotion.BakedDependencyHash : string.Empty).Append('|');
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
            MigrateLegacyIntent();
            var builder = new StringBuilder();
            CharacterMotionGenerationPlan plan = BuildGenerationPlan();
            for (int i = 0; i < plan.Runs.Length; i++)
            {
                CharacterMotionGenerationPart run = plan.Runs[i];
                CharacterMotionGenerationSettings s = overrideGenerationSettings
                    ? generationSettings.WithDefaults()
                    : run.Settings;
                builder.Append(run.PartIndex).Append(':').Append(run.RepetitionIndex).Append(':')
                    .Append(run.Intent != null ? run.Intent.Prompt : string.Empty).Append('|')
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
            IReadOnlyList<CharacterMotionPart> effectiveParts = parts;
            if ((effectiveParts == null || effectiveParts.Count == 0) && intent != null)
                effectiveParts = new[] { new CharacterMotionPart(intent) };
            return CharacterMotionPlanner.Build(effectiveParts, internalOverlapFrames, entryOverlapFrames,
                previousMotion != null);
        }

        private CharacterMotionGenerationSettings GetPrimarySettings(ICharacterMotionIntent resolved)
        {
            if (parts != null && parts.Count > 0 && parts[0] != null && runtimeIntentMatches(resolved, parts[0].Intent))
                return parts[0].Settings;
            return resolved.Settings;

            static bool runtimeIntentMatches(ICharacterMotionIntent value, CharacterMotionIntent serialized) =>
                serialized != null && ReferenceEquals(value, serialized);
        }

        private void ValidatePreviousMotion(CharacterMotionValidationResult result)
        {
            if (previousMotion == null) return;
            if (previousMotion == this)
            {
                result.Error("previous_motion_self_reference", "Previous Motion cannot reference this CharacterMotion.");
                return;
            }
            if (previousMotion.BakedClip == null)
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

        private void MigrateLegacyIntent()
        {
            if (intent == null) return;
            if (parts == null) parts = new List<CharacterMotionPart>();
            if (parts.Count == 0) parts.Add(new CharacterMotionPart(intent));
            intent = null;
        }

        private void OnValidate()
        {
            MigrateLegacyIntent();
            entryOverlapFrames = Mathf.Clamp(entryOverlapFrames, 0, KimodoConditioning.DefaultFrameCount - 1);
            internalOverlapFrames = Mathf.Clamp(internalOverlapFrames, 0, KimodoConditioning.DefaultFrameCount - 1);
        }
    }
}
