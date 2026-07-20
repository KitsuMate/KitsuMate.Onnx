using System;
using System.Threading;
using UnityEngine;

namespace KitsuMate.Onnx.Motion
{
    public sealed class CharacterMotionRequest
    {
        public KimodoTextEmbedding Embedding;
        public KimodoConstraintSet Constraints;
        public KimodoGenerationRequest Generation;
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
