using System;
using System.Threading;
using UnityEngine;

namespace KitsuMate.Onnx.Motion
{
    public sealed class CharacterMotionRequest
    {
        public CharacterMotionSegment[] Segments;
        public KimodoConstraintSet Constraints;
        public KimodoHumanoidMotion PreviousMotion;
        public int EntryHistoryFrames = 5;
        public int HistoryFrames = 5;
        public IProgress<int> Progress;
    }

    public sealed class CharacterMotionSegment
    {
        public KimodoTextEmbedding Embedding { get; }
        public KimodoGenerationRequest Generation { get; }

        public CharacterMotionSegment(KimodoTextEmbedding embedding, KimodoGenerationRequest generation)
        {
            Embedding = embedding ?? throw new ArgumentNullException(nameof(embedding));
            Generation = generation ?? throw new ArgumentNullException(nameof(generation));
        }
    }

    public sealed class CharacterMotionResult
    {
        public KimodoHumanoidMotion Motion { get; }
        public ModelIdentity ModelIdentity { get; }
        public CharacterMotionResult(KimodoHumanoidMotion motion, ModelIdentity identity) { Motion = motion; ModelIdentity = identity; }
    }

    public abstract class CharacterMotionEngine : InferenceEngine<CharacterMotionRequest, CharacterMotionResult>
    {
        public abstract KimodoModelCapabilities ConstraintCapabilities { get; }
        public abstract ModelIdentity RequiredEmbeddingModelIdentity { get; }
    }

    public abstract class CharacterMotionEngineRuntime : InferenceEngineRuntime<CharacterMotionRequest, CharacterMotionResult> { }
}
