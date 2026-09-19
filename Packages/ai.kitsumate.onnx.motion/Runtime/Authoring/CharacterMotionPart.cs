using System;
using System.Collections.Generic;
using UnityEngine;

namespace KitsuMate.Onnx.Motion
{
    /// <summary>One intent-backed portion of a CharacterMotion timeline.</summary>
    [Serializable]
    public sealed class CharacterMotionPart
    {
        [SerializeField] private CharacterMotionIntent intent;
        [SerializeField, Min(1)] private int repetitions = 1;
        [SerializeField, Min(2f / 30f)] private float durationSeconds = 2f;
        [SerializeField] private bool overrideGenerationSettings;
        [SerializeField] private CharacterMotionGenerationSettings generationSettings;

        public CharacterMotionIntent Intent => intent;
        public int Repetitions => Mathf.Max(1, repetitions);
        public float DurationSeconds => durationSeconds;
        public int FrameCount
        {
            get
            {
                double frames = Math.Round((double)durationSeconds * 30, MidpointRounding.AwayFromZero);
                if (double.IsNaN(frames) || frames < 2 || frames > int.MaxValue)
                    throw new ArgumentOutOfRangeException(nameof(durationSeconds), "Duration must be finite and contain at least two frames.");
                return checked((int)frames);
            }
        }
        public bool OverridesGenerationSettings => overrideGenerationSettings;
        public CharacterMotionGenerationSettings Settings => overrideGenerationSettings
            ? generationSettings.WithDefaults()
            : intent != null ? intent.Settings : CharacterMotionGenerationSettings.Default;

        public CharacterMotionPart() => generationSettings = CharacterMotionGenerationSettings.Default;

        public CharacterMotionPart(CharacterMotionIntent value, int repeatCount = 1, float durationSeconds = 2f)
        {
            intent = value;
            repetitions = Mathf.Max(1, repeatCount);
            this.durationSeconds = durationSeconds;
            generationSettings = CharacterMotionGenerationSettings.Default;
        }

        public void Configure(CharacterMotionIntent value, int repeatCount = 1)
        {
            intent = value;
            repetitions = Mathf.Max(1, repeatCount);
        }

        public void ConfigureGenerationSettings(bool enabled, CharacterMotionGenerationSettings value)
        {
            overrideGenerationSettings = enabled;
            generationSettings = value.WithDefaults();
        }
    }

    /// <summary>Calculated duration information for the complete authored motion.</summary>
    public readonly struct CharacterMotionTimeline
    {
        public int SegmentCount { get; }
        public int FrameCount { get; }

        internal CharacterMotionTimeline(int segmentCount, int frameCount)
        {
            SegmentCount = segmentCount;
            FrameCount = frameCount;
        }
    }

    internal readonly struct CharacterMotionGenerationPart
    {
        public int PartIndex { get; }
        public int RepetitionIndex { get; }
        public int OutputStartFrame { get; }
        public int FrameCount { get; }
        public CharacterMotionIntent Intent { get; }
        public CharacterMotionGenerationSettings Settings { get; }

        public CharacterMotionGenerationPart(int partIndex, int repetitionIndex, int outputStartFrame,
            int frameCount, CharacterMotionIntent intent, CharacterMotionGenerationSettings settings)
        {
            PartIndex = partIndex;
            RepetitionIndex = repetitionIndex;
            OutputStartFrame = outputStartFrame;
            FrameCount = frameCount;
            Intent = intent;
            Settings = settings;
        }
    }

    internal sealed class CharacterMotionGenerationPlan
    {
        public CharacterMotionGenerationPart[] Runs { get; }
        public int OutputFrameCount { get; }

        public CharacterMotionGenerationPlan(CharacterMotionGenerationPart[] runs, int outputFrameCount)
        {
            Runs = runs ?? Array.Empty<CharacterMotionGenerationPart>();
            OutputFrameCount = outputFrameCount;
        }
    }

    internal static class CharacterMotionPlanner
    {
        public static CharacterMotionGenerationPlan Build(IReadOnlyList<CharacterMotionPart> parts)
        {

            var runs = new List<CharacterMotionGenerationPart>();
            int outputFrameCount = 0;
            if (parts != null)
            {
                for (int partIndex = 0; partIndex < parts.Count; partIndex++)
                {
                    CharacterMotionPart part = parts[partIndex];
                    if (part == null) continue;
                    for (int repetition = 0; repetition < part.Repetitions; repetition++)
                    {
                        runs.Add(new CharacterMotionGenerationPart(partIndex, repetition, outputFrameCount,
                            part.FrameCount, part.Intent, part.Settings));
                        outputFrameCount = checked(outputFrameCount + part.FrameCount);
                    }
                }
            }

            return new CharacterMotionGenerationPlan(runs.ToArray(), outputFrameCount);
        }

        internal static void ValidateOverlap(int frames)
        {
            if (frames < 1 || frames > 19)
                throw new ArgumentOutOfRangeException(nameof(frames), "History must contain 1 to 19 frames.");
        }

        internal static int NextWindowFrames(int remaining, int history)
        {
            int capacity = Kimodo.KimodoTensorContract.MaxFrames - history;
            int windows = 1 + (remaining - 1) / capacity;
            return (int)(((long)remaining + windows - 1) / windows);
        }
    }
}
