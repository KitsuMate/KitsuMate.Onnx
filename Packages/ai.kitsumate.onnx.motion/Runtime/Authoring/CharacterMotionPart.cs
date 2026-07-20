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
        [SerializeField] private bool overrideGenerationSettings;
        [SerializeField] private CharacterMotionGenerationSettings generationSettings;

        public CharacterMotionIntent Intent => intent;
        public int Repetitions => Mathf.Max(1, repetitions);
        public bool OverridesGenerationSettings => overrideGenerationSettings;
        public CharacterMotionGenerationSettings Settings => overrideGenerationSettings
            ? generationSettings.WithDefaults()
            : intent != null ? intent.Settings : CharacterMotionGenerationSettings.Default;

        public CharacterMotionPart() => generationSettings = CharacterMotionGenerationSettings.Default;

        public CharacterMotionPart(CharacterMotionIntent value, int repeatCount = 1)
        {
            intent = value;
            repetitions = Mathf.Max(1, repeatCount);
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
        public int RunCount { get; }
        public int FrameCount { get; }
        public int EffectiveChainedFrameCount { get; }

        internal CharacterMotionTimeline(int runCount, int frameCount, int effectiveChainedFrameCount)
        {
            RunCount = runCount;
            FrameCount = frameCount;
            EffectiveChainedFrameCount = effectiveChainedFrameCount;
        }
    }

    internal readonly struct CharacterMotionGenerationPart
    {
        public int PartIndex { get; }
        public int RepetitionIndex { get; }
        public int OutputStartFrame { get; }
        public int LeadingOverlapFrames { get; }
        public CharacterMotionIntent Intent { get; }
        public CharacterMotionGenerationSettings Settings { get; }

        public CharacterMotionGenerationPart(int partIndex, int repetitionIndex, int outputStartFrame,
            int leadingOverlapFrames, CharacterMotionIntent intent, CharacterMotionGenerationSettings settings)
        {
            PartIndex = partIndex;
            RepetitionIndex = repetitionIndex;
            OutputStartFrame = outputStartFrame;
            LeadingOverlapFrames = leadingOverlapFrames;
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
        public static CharacterMotionGenerationPlan Build(IReadOnlyList<CharacterMotionPart> parts,
            int internalOverlapFrames, int entryOverlapFrames, bool hasPreviousMotion)
        {
            int framesPerRun = KimodoConditioning.DefaultFrameCount;
            if ((uint)internalOverlapFrames >= framesPerRun)
                throw new ArgumentOutOfRangeException(nameof(internalOverlapFrames),
                    $"Internal overlap must be between 0 and {framesPerRun - 1} frames.");
            if ((uint)entryOverlapFrames >= framesPerRun)
                throw new ArgumentOutOfRangeException(nameof(entryOverlapFrames),
                    $"Entry overlap must be between 0 and {framesPerRun - 1} frames.");

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
                        int leadingOverlap = runs.Count == 0
                            ? hasPreviousMotion ? entryOverlapFrames : 0
                            : internalOverlapFrames;
                        int outputStart = runs.Count == 0 ? 0 : outputFrameCount - internalOverlapFrames;
                        runs.Add(new CharacterMotionGenerationPart(partIndex, repetition, outputStart,
                            leadingOverlap, part.Intent, part.Settings));
                        outputFrameCount = outputStart + framesPerRun;
                    }
                }
            }

            return new CharacterMotionGenerationPlan(runs.ToArray(), outputFrameCount);
        }
    }
}
