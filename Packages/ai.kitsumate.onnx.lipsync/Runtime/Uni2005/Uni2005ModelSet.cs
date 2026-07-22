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
        [SerializeField] private OnnxModelReference _acousticModelSource = new();
        [Header("Vocabulary")]
        [SerializeField, Tooltip("Phone vocabulary JSON file")]
        private TextAsset _vocabulary;
        
        [Header("Model Info")]
        [SerializeField, Tooltip("Model variant description")]
        private string _variantDescription = "Allosaurus Universal Phone Recognizer";
        
        /// <summary>Acoustic model for phoneme detection.</summary>
        public OnnxModelReference AcousticModel => _acousticModelSource;
        
        /// <summary>Vocabulary JSON file.</summary>
        public TextAsset Vocabulary => _vocabulary;
        
        public override string DisplayName => 
            string.IsNullOrEmpty(_variantDescription) 
                ? "Uni2005" 
                : $"Uni2005 ({_variantDescription})";
        
        public override bool IsComplete => 
            _acousticModelSource.IsAvailable && _vocabulary != null;
        
        public override IOnnxModelSource[] GetAllModels()
        {
            return new IOnnxModelSource[] { _acousticModelSource };
        }
        
        public override ModelValidationResult Validate(ModelValidationContext context)
        {
            var result = new ModelValidationResult();
            if (!_acousticModelSource.IsAvailable) result.Error("missing_model", "Acoustic model is not available.");
            if (_vocabulary == null) result.Error("missing_vocabulary", "Vocabulary file is not assigned.");
            RequireInput(_acousticModelSource, "mfcc", result);
            RequireSchema(_acousticModelSource, result);
            return ValidateCommon(context, result);
        }
        
#if UNITY_EDITOR
        public void SetVocabulary(TextAsset vocabulary)
        {
            _vocabulary = vocabulary;
            UnityEditor.EditorUtility.SetDirty(this);
        }

        public void SetModels(OnnxModelAsset acousticModel, TextAsset vocabulary)
        {
            _acousticModelSource.ConfigureAsset(acousticModel);
            _vocabulary = vocabulary;
            UnityEditor.EditorUtility.SetDirty(this);
        }
#endif
    }
}
