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
        [SerializeField] private Transform leftHand;
        [SerializeField] private Transform rightHand;
        [SerializeField] private Transform leftFoot;
        [SerializeField] private Transform rightFoot;

        public int Frame => frame;
        public CharacterMotionConstraintType Constraints => constraints;
        public CharacterMotionPoseSnapshot Pose => pose;
        public Transform LeftHand => leftHand;
        public Transform RightHand => rightHand;
        public Transform LeftFoot => leftFoot;
        public Transform RightFoot => rightFoot;
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
            switch (type)
            {
                case CharacterMotionConstraintType.LeftHand: return leftHand;
                case CharacterMotionConstraintType.RightHand: return rightHand;
                case CharacterMotionConstraintType.LeftFoot: return leftFoot;
                case CharacterMotionConstraintType.RightFoot: return rightFoot;
                default: throw new ArgumentOutOfRangeException(nameof(type));
            }
        }

        public void SetEffector(CharacterMotionConstraintType type, Transform value)
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

        private void OnDrawGizmos()
        {
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireSphere(transform.position, 0.035f);
            if ((constraints & CharacterMotionConstraintType.RootHeading) != 0)
                Gizmos.DrawLine(transform.position, transform.position + transform.forward * 0.25f);
            DrawEffector(leftHand, new Color(0.3f, 0.7f, 1f));
            DrawEffector(rightHand, new Color(0.3f, 0.7f, 1f));
            DrawEffector(leftFoot, new Color(1f, 0.65f, 0.2f));
            DrawEffector(rightFoot, new Color(1f, 0.65f, 0.2f));
        }

        private void DrawEffector(Transform target, Color color)
        {
            if (target == null) return;
            Gizmos.color = color;
            Gizmos.DrawLine(transform.position, target.position);
            Gizmos.DrawWireSphere(target.position, 0.025f);
        }
    }
}
