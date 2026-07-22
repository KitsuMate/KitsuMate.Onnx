using UnityEngine;

namespace KitsuMate.Onnx.Motion
{
    public static class CharacterMotionSpace
    {
        public static Vector3 WorldToCanonical(Vector3 world, Transform origin)
        {
            Quaternion inverse = Quaternion.Inverse(origin != null ? origin.rotation : Quaternion.identity);
            Vector3 local = inverse * (world - (origin != null ? origin.position : Vector3.zero));
            return new Vector3(-local.x, local.y, local.z);
        }

        public static Quaternion WorldToCanonical(Quaternion world, Transform origin)
        {
            Quaternion local = Quaternion.Inverse(origin != null ? origin.rotation : Quaternion.identity) * world;
            return ReflectX(local);
        }

        public static Vector2 WorldForwardToHeading(Vector3 forward, Transform origin)
        {
            Vector3 local = Quaternion.Inverse(origin != null ? origin.rotation : Quaternion.identity) * forward;
            local.y = 0f;
            if (local.sqrMagnitude < 1e-10f) return Vector2.zero;
            local.Normalize();
            return new Vector2(local.z, -local.x);
        }

        public static Quaternion ReflectX(Quaternion value)
        {
            Quaternion result = new Quaternion(value.x, -value.y, -value.z, value.w);
            float magnitude = Mathf.Sqrt(result.x * result.x + result.y * result.y + result.z * result.z + result.w * result.w);
            return magnitude > 1e-8f
                ? new Quaternion(result.x / magnitude, result.y / magnitude, result.z / magnitude, result.w / magnitude)
                : Quaternion.identity;
        }

        public static bool IsFinite(Vector3 value) => IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
        public static bool IsFinite(Quaternion value) => IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z) && IsFinite(value.w);
        private static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
    }
}
