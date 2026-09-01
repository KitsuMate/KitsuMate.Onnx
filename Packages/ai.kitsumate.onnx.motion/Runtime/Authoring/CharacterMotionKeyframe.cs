using System;
using UnityEngine;

namespace KitsuMate.Onnx.Motion
{
    [Flags]
    public enum CharacterMotionConstraintType
    {
        None = 0,
        RootPosition = 1 << 0,
        RootHeading = 1 << 1,
        FullBodyPose = 1 << 2,
        LeftHand = 1 << 3,
        RightHand = 1 << 4,
        LeftFoot = 1 << 5,
        RightFoot = 1 << 6,
    }

    [DisallowMultipleComponent]
    public sealed class CharacterMotionKeyframe : MonoBehaviour
    {
        [SerializeField, Min(0)] private int frame;
        [SerializeField] private CharacterMotionConstraintType constraints = CharacterMotionConstraintType.RootPosition;
        [SerializeField] private CharacterMotionPoseSnapshot pose = new CharacterMotionPoseSnapshot();
        [NonSerialized] private Transform leftHand;
        [NonSerialized] private Transform rightHand;
        [NonSerialized] private Transform leftFoot;
        [NonSerialized] private Transform rightFoot;

        public int Frame => frame;
        public CharacterMotionConstraintType Constraints => constraints;
        public CharacterMotionPoseSnapshot Pose => pose;
        public Transform LeftHand => GetEffector(CharacterMotionConstraintType.LeftHand);
        public Transform RightHand => GetEffector(CharacterMotionConstraintType.RightHand);
        public Transform LeftFoot => GetEffector(CharacterMotionConstraintType.LeftFoot);
        public Transform RightFoot => GetEffector(CharacterMotionConstraintType.RightFoot);
        public bool RequiresPose => (constraints & (CharacterMotionConstraintType.FullBodyPose |
            CharacterMotionConstraintType.LeftHand | CharacterMotionConstraintType.RightHand |
            CharacterMotionConstraintType.LeftFoot | CharacterMotionConstraintType.RightFoot)) != 0;

        public void Configure(int frameIndex, CharacterMotionConstraintType constraintTypes)
        {
            if (frameIndex < 0) throw new ArgumentOutOfRangeException(nameof(frameIndex));
            frame = frameIndex;
            constraints = constraintTypes;
        }

        public Transform GetEffector(CharacterMotionConstraintType type)
        {
            Transform cached = type switch
            {
                CharacterMotionConstraintType.LeftHand => leftHand,
                CharacterMotionConstraintType.RightHand => rightHand,
                CharacterMotionConstraintType.LeftFoot => leftFoot,
                CharacterMotionConstraintType.RightFoot => rightFoot,
                _ => throw new ArgumentOutOfRangeException(nameof(type)),
            };
            if (IsNamedDirectChild(cached, type)) return cached;

            cached = FindNamedDirectChild(type);
            SetEffectorCache(type, cached);
            return cached;
        }

        public void SetEffector(CharacterMotionConstraintType type, Transform value)
        {
            string controlName = GetEffectorName(type);
            if (value != null && (value.parent != transform || value.name != controlName))
                throw new ArgumentException(
                    $"The {controlName} effector must be a direct child named '{controlName}'.", nameof(value));
            SetEffectorCache(type, value);
        }

        private Transform FindNamedDirectChild(CharacterMotionConstraintType type)
        {
            string controlName = GetEffectorName(type);
            for (int i = 0; i < transform.childCount; i++)
            {
                Transform child = transform.GetChild(i);
                if (child.name == controlName) return child;
            }
            return null;
        }

        private bool IsNamedDirectChild(Transform value, CharacterMotionConstraintType type)
            => value != null && value.parent == transform && value.name == GetEffectorName(type);

        private void SetEffectorCache(CharacterMotionConstraintType type, Transform value)
        {
            switch (type)
            {
                case CharacterMotionConstraintType.LeftHand: leftHand = value; break;
                case CharacterMotionConstraintType.RightHand: rightHand = value; break;
                case CharacterMotionConstraintType.LeftFoot: leftFoot = value; break;
                case CharacterMotionConstraintType.RightFoot: rightFoot = value; break;
                default: throw new ArgumentOutOfRangeException(nameof(type));
            }
        }

        private static string GetEffectorName(CharacterMotionConstraintType type) => type switch
        {
            CharacterMotionConstraintType.LeftHand => nameof(CharacterMotionConstraintType.LeftHand),
            CharacterMotionConstraintType.RightHand => nameof(CharacterMotionConstraintType.RightHand),
            CharacterMotionConstraintType.LeftFoot => nameof(CharacterMotionConstraintType.LeftFoot),
            CharacterMotionConstraintType.RightFoot => nameof(CharacterMotionConstraintType.RightFoot),
            _ => throw new ArgumentOutOfRangeException(nameof(type)),
        };

    }
}
