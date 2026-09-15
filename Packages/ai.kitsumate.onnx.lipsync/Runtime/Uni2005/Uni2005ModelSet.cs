using KitsuMate.Onnx.Download;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using UnityEngine;
using KitsuMate.Onnx;

namespace KitsuMate.Onnx.LipSync.Uni2005
{
    /// <summary>
    /// Groups Uni2005 Allosaurus ONNX model with vocabulary.
    /// </summary>
    [CreateAssetMenu(fileName = "Uni2005ModelSet", menuName = "KitsuMate/ONNX/LipSync/Uni2005 Model Set")]
    public class Uni2005ModelSet : StandardModelSet
    {
        public override string[] DownloadCompanionRoles => new[] { "vocabulary" };

        public override TextFileReference[] GetAllTextFiles() => new[] { _vocabulary };

        public override System.Collections.Generic.IEnumerable<(string Family, string Repository)> RepositorySuggestions
        {
            get
            {
                yield return ("uni2005", "KitsuMate/uni2005-onnx");
            }
        }

        protected override Task BindInstallationAsync(DownloadedModel installation, ResolvedModelSet resources, CancellationToken cancellationToken)
        {
            installation.ConfigureModel(_acousticModelSource, "model");
            _vocabulary = resources.ReadText(installation, "vocabulary");
            return Task.CompletedTask;
        }

        [SerializeField] private OnnxModelReference _acousticModelSource = new();
        [Header("Vocabulary")]
        [SerializeField, Tooltip("Phone vocabulary JSON file")]
        private TextFileReference _vocabulary = new();
        
        /// <summary>Acoustic model for phoneme detection.</summary>
        public OnnxModelReference AcousticModel => _acousticModelSource;
        
        /// <summary>Vocabulary JSON file.</summary>
        public TextFileReference Vocabulary => _vocabulary?.IsAvailable == true ? _vocabulary : null;
        
        public override string DisplayName => string.IsNullOrWhiteSpace(name) ? "Uni2005" : name;
        
        public override bool IsComplete => 
            _acousticModelSource.IsAvailable && _vocabulary?.IsAvailable == true;
        
        public override IOnnxModelSource[] GetAllModels()
        {
            return new IOnnxModelSource[] { _acousticModelSource };
        }
        
        public override ModelValidationResult Validate(ModelValidationContext context)
        {
            var result = new ModelValidationResult();
            if (!_acousticModelSource.IsAvailable) result.Error("missing_model", "Acoustic model is not available.");
            if (_vocabulary?.IsAvailable != true) result.Error("missing_vocabulary", "Vocabulary file is not assigned.");
            RequireInput(_acousticModelSource, "mfcc", result);
            RequireSchema(_acousticModelSource, result);
            return ValidateCommon(context, result);
        }
        
#if UNITY_EDITOR
        public void SetVocabulary(TextFileReference vocabulary)
        {
            _vocabulary = vocabulary;
            UnityEditor.EditorUtility.SetDirty(this);
        }

        public void SetModels(OnnxModelAsset acousticModel, TextFileReference vocabulary)
        {
            _acousticModelSource.ConfigureAsset(acousticModel);
            _vocabulary = vocabulary;
            UnityEditor.EditorUtility.SetDirty(this);
        }
#endif
    }
}
