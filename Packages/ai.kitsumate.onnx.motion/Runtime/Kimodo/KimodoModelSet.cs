using KitsuMate.Onnx.Download;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Kimodo
{
    [CreateAssetMenu(fileName = "KimodoModelSet", menuName = "KitsuMate/ONNX/Motion/Kimodo Model Set")]
    public sealed class KimodoModelSet : StandardModelSet
    {
        public override System.Collections.Generic.IEnumerable<(string Family, string Repository)> RepositorySuggestions
        {
            get
            {
                yield return ("kimodo", "KitsuMate/Kimodo-SOMA-RP-v1.1-ONNX");
            }
        }

        protected override Task BindInstallationAsync(DownloadedModel installation, ResolvedModelSet resources, CancellationToken cancellationToken)
        {
            installation.ConfigureModel(motionModel, "model");
            return Task.CompletedTask;
        }

        [SerializeField] private OnnxModelReference motionModel = new();
        [SerializeField, Tooltip("Embedding model whose output contract this Kimodo model expects.")]
        private ModelSet requiredEmbeddingModelSet;

        public OnnxModelReference MotionModel => motionModel;
        public ModelSet RequiredEmbeddingModelSet => requiredEmbeddingModelSet;
        public ModelIdentity RequiredEmbeddingModelIdentity => requiredEmbeddingModelSet != null
            ? requiredEmbeddingModelSet.Identity
            : default;
        public override string DisplayName => string.IsNullOrWhiteSpace(name) ? "Kimodo" : name;
        public override bool IsComplete => motionModel != null && motionModel.IsAvailable &&
            requiredEmbeddingModelSet != null;
        public override IOnnxModelSource[] GetAllModels() => motionModel != null ? new IOnnxModelSource[] { motionModel } : System.Array.Empty<IOnnxModelSource>();

#if UNITY_EDITOR
        public void SetRequiredEmbedding(ModelSet value)
        {
            requiredEmbeddingModelSet = value;
            UnityEditor.EditorUtility.SetDirty(this);
        }
#endif
        public override ModelValidationResult Validate(ModelValidationContext context)
        {
            var result = new ModelValidationResult();
            if (motionModel == null) result.Error("missing_model", "Kimodo motion model is not assigned.");
            if (requiredEmbeddingModelSet == null)
                result.Error("missing_embedding_contract", "Assign the embedding model required by this Kimodo model.");
            foreach (string input in new[] { "motion", "motion_valid", "text_embedding", "timestep", KimodoTensorContract.FirstHeading, "constraint_mask", "observed_motion" }) RequireInput(motionModel, input, result);
            RequireOutput(motionModel, "predicted_clean_motion", result);
            if (motionModel != null && motionModel.HasInspectedMetadata)
            {
                foreach (var input in motionModel.Inputs)
                {
                    if (input.Name != "motion" && input.Name != "motion_valid" &&
                        input.Name != "constraint_mask" && input.Name != "observed_motion") continue;
                    int rank = input.Name == "motion_valid" ? 2 : 3;
                    if (input.shape == null || input.shape.Length != rank || input.shape[0] != 3 || input.shape[1] >= 0 ||
                        rank == 3 && input.shape[2] != 369)
                        result.Error("invalid_motion_shape", $"{input.Name} must have shape [3,T{(rank == 3 ? ",369" : "")}], with dynamic T.");
                }
                foreach (var output in motionModel.Outputs)
                    if (output.Name == "predicted_clean_motion" &&
                        (output.shape == null || output.shape.Length != 3 || output.shape[0] != 3 ||
                         output.shape[1] >= 0 || output.shape[2] != 369))
                        result.Error("invalid_motion_shape", "predicted_clean_motion must have shape [3,T,369], with dynamic T.");
            }
            return ValidateCommon(context, result);
        }
    }
}
