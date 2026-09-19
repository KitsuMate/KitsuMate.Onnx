using System;
using System.Threading;
using System.Diagnostics;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Kimodo
{
    /// <summary>Generates bounded windows in canonical space, retaining history only once in the output.</summary>
    internal static class KimodoMotionSequence
    {
        internal static KimodoHumanoidMotion Generate(CharacterMotionRequest request,
            Func<KimodoTextEmbedding, KimodoConditioning, KimodoGenerationRequest, CancellationToken, float[]> generate,
            CancellationToken cancellationToken, bool diagnostics = false)
        {
            if (request?.Segments == null || request.Segments.Length == 0)
                throw new ArgumentException("Motion request requires at least one segment.", nameof(request));
            CharacterMotionPlanner.ValidateOverlap(request.HistoryFrames);
            CharacterMotionPlanner.ValidateOverlap(request.EntryHistoryFrames);
            int total = 0;
            foreach (var segment in request.Segments)
            {
                if (segment == null) throw new ArgumentException("Motion segments cannot be null.", nameof(request));
                total = checked(total + segment.Generation.FrameCount);
            }
            var compiler = new KimodoConstraintCompiler();
            KimodoConditioning timeline = compiler.Compile(request.Constraints, total);
            var output = new float[checked(total * KimodoConditioning.FeatureCount)];
            float[] previous = request.PreviousMotion?.SourceMotion;
            if (request.PreviousMotion != null && previous == null)
                throw new ArgumentException("Previous motion has no canonical source data.", nameof(request));
            var watch = Stopwatch.StartNew();
            int completed = 0, windowIndex = 0;
            foreach (CharacterMotionSegment segment in request.Segments)
            {
                int remaining = segment.Generation.FrameCount;
                while (remaining > 0)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int history = previous == null ? 0 : Math.Min(
                        completed == 0 ? request.EntryHistoryFrames : request.HistoryFrames,
                        previous.Length / KimodoConditioning.FeatureCount);
                    int fresh = CharacterMotionPlanner.NextWindowFrames(remaining, history);
                    int frames = history + fresh;
                    var values = new float[frames * KimodoConditioning.FeatureCount];
                    var mask = new bool[values.Length];
                    // Include authored constraints in an internal overlap; entry history precedes this timeline.
                    int timelineStart = Math.Max(0, completed - history);
                    int leading = Math.Max(0, history - completed);
                    int sourceCount = frames - leading;
                    timeline.ObservedMotion.Span.Slice(timelineStart * 369, sourceCount * 369)
                        .CopyTo(values.AsSpan(leading * 369));
                    timeline.MotionMask.Span.Slice(timelineStart * 369, sourceCount * 369)
                        .CopyTo(mask.AsSpan(leading * 369));

                    Vector2 translation = Vector2.zero;
                    float heading = segment.Generation.FirstHeadingRadians;
                    if (history > 0)
                    {
                        int start = previous.Length - history * 369;
                        translation = new Vector2(KimodoMotionProcessing.Value(previous, start),
                            KimodoMotionProcessing.Value(previous, start + 2));
                        heading = Mathf.Atan2(KimodoMotionProcessing.Value(previous, start + 4),
                            KimodoMotionProcessing.Value(previous, start + 3));
                        for (int frame = 0; frame < history; frame++)
                        {
                            int offset = frame * 369;
                            for (int feature = 0; feature < 369; feature++)
                            {
                                if (!IsHistoryFeature(feature)) continue;
                                float value = previous[start + offset + feature];
                                if (mask[offset + feature] && Mathf.Abs(values[offset + feature] - value) * KimodoSomaRuntimeData.Scale[feature] > .01f)
                                    throw new InvalidOperationException($"A constraint at frame {timelineStart + frame} conflicts with continuation history.");
                                values[offset + feature] = value;
                                mask[offset + feature] = true;
                            }
                        }
                    }
                    Translate(values, -translation, mask);
                    var conditioning = new KimodoConditioning(values, mask, frames, copy: false);
                    var settings = segment.Generation;
                    var window = new KimodoGenerationRequest(unchecked(settings.Seed + windowIndex), settings.DenoisingSteps,
                        settings.TextGuidance, settings.ConstraintGuidance, heading, frames);
                    float[] motion = generate(segment.Embedding, conditioning, window, cancellationToken);
                    if (motion == null || motion.Length != frames * 369)
                        throw new InvalidOperationException("Generator returned the wrong motion shape.");
                    KimodoMotionProcessing.Correct(motion, conditioning, cancellationToken);
                    Translate(motion, translation);
                    // Already published history is immutable. The next sample follows its conditioned tail.
                    Array.Copy(motion, history * 369, output, completed * 369, fresh * 369);
                    previous = motion;
                    completed += fresh;
                    remaining -= fresh;
                    windowIndex++;
                    request.Progress?.Report(completed);
                }
            }
            long generationMilliseconds = watch.ElapsedMilliseconds;
            cancellationToken.ThrowIfCancellationRequested();
            var result = KimodoMotionDecoder.Decode(output, total, null);
            if (diagnostics) result.Diagnostics = new KimodoGenerationDiagnostics(
                generationMilliseconds, watch.ElapsedMilliseconds - generationMilliseconds, output);
            result.SetSourceMotion(output);
            return result;
        }

        private static bool IsHistoryFeature(int feature) => feature < 95 ||
            feature >= 95 + 13 * 6 && feature < 95 + 14 * 6 ||
            feature >= 95 + 19 * 6 && feature < 95 + 20 * 6 ||
            feature >= 95 + 24 * 6 && feature < 95 + 25 * 6 ||
            feature >= 95 + 28 * 6 && feature < 95 + 29 * 6;

        internal static void Translate(float[] values, Vector2 delta, bool[] mask = null)
        {
            for (int offset = 0; offset < values.Length; offset += 369)
            {
                if (mask == null || mask[offset]) values[offset] += delta.x / KimodoSomaRuntimeData.Scale[0];
                if (mask == null || mask[offset + 2]) values[offset + 2] += delta.y / KimodoSomaRuntimeData.Scale[2];
            }
        }
    }
}
