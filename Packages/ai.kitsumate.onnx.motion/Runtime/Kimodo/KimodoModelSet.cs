using UnityEngine;

namespace KitsuMate.Onnx.Motion.Kimodo
{
    [CreateAssetMenu(fileName = "KimodoModelSet", menuName = "KitsuMate/ONNX/Motion/Kimodo Model Set")]
    public sealed class KimodoModelSet : StandardModelSet
    {
        [SerializeField] private OnnxModelReference motionModel = new();
        [SerializeField, Tooltip("Embedding model whose output contract this Kimodo variant expects.")]
        private ModelSet requiredEmbeddingModelSet;

        public OnnxModelReference MotionModel => motionModel;
        public ModelSet RequiredEmbeddingModelSet => requiredEmbeddingModelSet;
        public ModelIdentity RequiredEmbeddingModelIdentity => requiredEmbeddingModelSet != null
            ? requiredEmbeddingModelSet.Identity
            : default;
        public override string DisplayName => string.IsNullOrWhiteSpace(name) ? "Kimodo" : name;
        public override bool IsComplete => motionModel != null && motionModel.IsAvailable &&
            requiredEmbeddingModelSet != null && requiredEmbeddingModelSet.IsComplete;
        public override IOnnxModelSource[] GetAllModels() => motionModel != null ? new IOnnxModelSource[] { motionModel } : System.Array.Empty<IOnnxModelSource>();
        public override ModelValidationResult Validate(ModelValidationContext context)
        {
            var result = new ModelValidationResult();
            if (motionModel == null) result.Error("missing_model", "Kimodo motion model is not assigned.");
            if (requiredEmbeddingModelSet == null)
                result.Error("missing_embedding_contract", "Assign the embedding model required by this Kimodo variant.");
            else if (!requiredEmbeddingModelSet.IsComplete)
                result.Error("incomplete_embedding_contract", $"Required embedding model set '{requiredEmbeddingModelSet.name}' is incomplete.");
            foreach (string input in new[] { "motion", "motion_valid", "text_embedding", "timestep", KimodoTensorContract.FirstHeading, "constraint_mask", "observed_motion" }) RequireInput(motionModel, input, result);
            RequireOutput(motionModel, "predicted_clean_motion", result);
            return ValidateCommon(context, result);
        }
    }
}
