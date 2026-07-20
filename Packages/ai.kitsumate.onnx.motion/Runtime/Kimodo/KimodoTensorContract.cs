namespace KitsuMate.Onnx.Motion.Kimodo
{
    internal static class KimodoTensorContract
    {
        public const int Batch = 3;
        public const int Frames = 60;
        public const int MotionDimension = 369;
        public const int TextTokens = 50;
        public const int TextDimension = 4096;
        public const int JointCount = 30;
        public const float FramesPerSecond = 30f;

        public const string Motion = "motion";
        public const string MotionValid = "motion_valid";
        public const string TextEmbedding = "text_embedding";
        public const string Timestep = "timestep";
        public const string FirstHeading = "first_heading_angle";
        public const string ConstraintMask = "constraint_mask";
        public const string ObservedMotion = "observed_motion";
        public const string PredictedCleanMotion = "predicted_clean_motion";
    }
}
