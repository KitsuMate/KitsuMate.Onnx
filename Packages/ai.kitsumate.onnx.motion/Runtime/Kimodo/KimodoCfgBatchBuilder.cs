using System;

namespace KitsuMate.Onnx.Motion.Kimodo
{
    internal static class KimodoCfgBatchBuilder
    {
        internal static void Build(
            KimodoTextEmbedding embedding,
            KimodoConditioning conditioning,
            float[] textBatch,
            bool[] constraintMaskBatch,
            float[] observedMotionBatch)
        {
            if (embedding == null) throw new ArgumentNullException(nameof(embedding));
            if (conditioning == null) throw new ArgumentNullException(nameof(conditioning));
            int motionLength = KimodoTensorContract.Frames * KimodoTensorContract.MotionDimension;
            int textLength = KimodoTensorContract.Batch * KimodoTensorContract.TextTokens * KimodoTensorContract.TextDimension;
            if (textBatch == null || textBatch.Length != textLength) throw new ArgumentException("Text CFG batch has the wrong shape.", nameof(textBatch));
            if (constraintMaskBatch == null || constraintMaskBatch.Length != KimodoTensorContract.Batch * motionLength)
                throw new ArgumentException("Constraint-mask CFG batch has the wrong shape.", nameof(constraintMaskBatch));
            if (observedMotionBatch == null || observedMotionBatch.Length != KimodoTensorContract.Batch * motionLength)
                throw new ArgumentException("Observed-motion CFG batch has the wrong shape.", nameof(observedMotionBatch));

            Array.Clear(textBatch, 0, textBatch.Length);
            Array.Clear(constraintMaskBatch, 0, constraintMaskBatch.Length);
            Array.Clear(observedMotionBatch, 0, observedMotionBatch.Length);
            embedding.Values.Span.CopyTo(textBatch.AsSpan(0, KimodoTensorContract.TextDimension));
            conditioning.MotionMask.Span.CopyTo(constraintMaskBatch.AsSpan(motionLength, motionLength));
            ReadOnlySpan<float> observed = conditioning.ObservedMotion.Span;
            observed.CopyTo(observedMotionBatch.AsSpan(0, motionLength));
            observed.CopyTo(observedMotionBatch.AsSpan(motionLength, motionLength));
            observed.CopyTo(observedMotionBatch.AsSpan(motionLength * 2, motionLength));
        }
    }
}
