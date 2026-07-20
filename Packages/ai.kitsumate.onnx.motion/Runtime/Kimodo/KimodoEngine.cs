using System;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Kimodo
{
    [CreateAssetMenu(fileName = "KimodoEngine", menuName = "KitsuMate/ONNX/Motion/Kimodo Engine")]
    public sealed class KimodoEngine : CharacterMotionEngine
    {
        [SerializeField] private KimodoModelSet modelSet;
        [SerializeField] private bool diagnostics;
        public override ModelSet ModelSet => modelSet;
        public override KimodoModelCapabilities ConstraintCapabilities => KimodoConstraintCompiler.SomaRpV11Capabilities;
        public override ModelIdentity RequiredEmbeddingModelIdentity => modelSet != null
            ? modelSet.RequiredEmbeddingModelIdentity
            : default;
        protected override InferenceEngineRuntime<CharacterMotionRequest, CharacterMotionResult> CreateRuntime() => new KimodoEngineRuntime(modelSet, diagnostics);
    }

    public sealed class KimodoEngineRuntime : CharacterMotionEngineRuntime
    {
        private readonly KimodoModelSet modelSet;
        private readonly bool diagnostics;
        private KimodoMotionGenerator generator;
        public KimodoEngineRuntime(KimodoModelSet modelSet, bool diagnostics) { this.modelSet = modelSet; this.diagnostics = diagnostics; }
        protected override Task OnLoadAsync(CancellationToken cancellationToken)
        {
            return Task.Run(() =>
            {
                generator = new KimodoMotionGenerator(modelSet.MotionModel, Backend, diagnostics);
                generator.Initialize(cancellationToken);
            }, cancellationToken);
        }
        protected override Task<CharacterMotionResult> OnRunAsync(CharacterMotionRequest request, CancellationToken cancellationToken)
        {
            if (request?.Embedding == null) throw new ArgumentException("Motion request requires an embedding.", nameof(request));
            return Task.Run(() =>
            {
                KimodoGenerationRequest generation = request.Generation ?? new KimodoGenerationRequest();
                KimodoConditioning conditioning = new KimodoConstraintCompiler().Compile(request.Constraints ?? KimodoConstraintSet.Empty);
                KimodoHumanoidMotion motion = generator.Generate(request.Embedding, conditioning, generation, cancellationToken);
                return new CharacterMotionResult(motion, modelSet.Identity);
            }, cancellationToken);
        }
        protected override Task OnUnloadAsync(CancellationToken cancellationToken) { generator?.Dispose(); generator = null; return Task.CompletedTask; }
        protected override void OnDispose() { generator?.Dispose(); generator = null; }
    }
}
