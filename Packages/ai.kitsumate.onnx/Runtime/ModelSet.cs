using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace KitsuMate.Onnx
{
    [Serializable]
    public readonly struct ModelIdentity : IEquatable<ModelIdentity>
    {
        public readonly string Family, ModelId, Revision, Variant, ContentHash;
        public ModelIdentity(string family, string modelId, string revision, string variant, string contentHash)
        { Family = family ?? ""; ModelId = modelId ?? ""; Revision = revision ?? ""; Variant = variant ?? ""; ContentHash = contentHash ?? ""; }
        public bool Equals(ModelIdentity other) => Family == other.Family && ModelId == other.ModelId && Revision == other.Revision && Variant == other.Variant && ContentHash == other.ContentHash;
        public override bool Equals(object obj) => obj is ModelIdentity other && Equals(other);
        public override int GetHashCode() => HashCode.Combine(Family, ModelId, Revision, Variant, ContentHash);
        public override string ToString() => $"{Family}/{ModelId}@{Revision}:{Variant}#{ContentHash}";
    }

    [Serializable]
    public struct ModelCapabilities
    {
        [SerializeField] private string[] values;
        public ModelCapabilities(IEnumerable<string> values) { this.values = values?.ToArray() ?? Array.Empty<string>(); }
        public IReadOnlyList<string> Values => values ?? Array.Empty<string>();
        public bool Contains(string value) => Values.Contains(value);
    }
    public enum ModelDiagnosticSeverity { Warning, Error }
    public readonly struct ModelDiagnostic { public readonly ModelDiagnosticSeverity Severity; public readonly string Code, Message; public ModelDiagnostic(ModelDiagnosticSeverity severity, string code, string message) { Severity = severity; Code = code; Message = message; } public override string ToString() => Message; }
    public sealed class ModelValidationResult
    {
        private readonly List<ModelDiagnostic> diagnostics = new();
        public IReadOnlyList<ModelDiagnostic> Diagnostics => diagnostics;
        public bool IsValid => diagnostics.All(x => x.Severity != ModelDiagnosticSeverity.Error);
        public void Error(string code, string message) => diagnostics.Add(new ModelDiagnostic(ModelDiagnosticSeverity.Error, code, message));
        public void Warning(string code, string message) => diagnostics.Add(new ModelDiagnostic(ModelDiagnosticSeverity.Warning, code, message));
    }
    public readonly struct ModelValidationContext { public readonly OnnxBackend Backend; public ModelValidationContext(OnnxBackend backend) { Backend = backend; } }
    public sealed class ModelValidationException : InvalidOperationException
    {
        public ModelValidationResult Validation { get; }
        public ModelValidationException(ModelValidationResult validation) : base(string.Join("\n", validation.Diagnostics.Select(x => x.Message))) { Validation = validation; }
    }

    public abstract class ModelSet : ScriptableObject
    {
        public abstract string DisplayName { get; }
        public abstract ModelIdentity Identity { get; }
        public abstract ModelCapabilities Capabilities { get; }
        public abstract bool IsComplete { get; }
        public abstract IOnnxModelSource[] GetAllModels();
        public abstract ModelValidationResult Validate(ModelValidationContext context);
    }

    public abstract class StandardModelSet : ModelSet
    {
        [Header("Model Identity")]
        [SerializeField] private string family;
        [SerializeField] private string modelId;
        [SerializeField] private string revision = "main";
        [SerializeField] private string variant = "fp32";
        [SerializeField] private string contentHash;
        [SerializeField] private ModelCapabilities capabilities;

        protected virtual string DefaultFamily => GetType().Namespace ?? "onnx";
        protected virtual string DefaultModelId => GetType().Name;
        public override ModelIdentity Identity => new(string.IsNullOrWhiteSpace(family) ? DefaultFamily : family, string.IsNullOrWhiteSpace(modelId) ? DefaultModelId : modelId, revision, variant, contentHash);
        public override ModelCapabilities Capabilities => capabilities;

#if UNITY_EDITOR
        public void SetDownloadMetadata(ModelIdentity identity, IEnumerable<string> downloadedCapabilities)
        {
            family = identity.Family;
            modelId = identity.ModelId;
            revision = identity.Revision;
            variant = identity.Variant;
            contentHash = identity.ContentHash;
            capabilities = new ModelCapabilities(downloadedCapabilities);
            UnityEditor.EditorUtility.SetDirty(this);
        }
#endif

        protected ModelValidationResult ValidateCommon(ModelValidationContext context, ModelValidationResult result = null)
        {
            result ??= new ModelValidationResult();
            foreach (IOnnxModelSource model in GetAllModels())
                if (model != null && !model.IsAvailable) result.Error("unresolved_model", $"Model '{model.SourceName}' has no resolvable data.");
            return result;
        }

        protected static void RequireInput(IOnnxModelSource model, string name, ModelValidationResult result)
        {
            if (model == null || !model.HasInspectedMetadata) return;
            if (!model.Inputs.Any(x => x.Name == name)) result.Error("missing_input", $"Model '{model.SourceName}' is missing input '{name}'.");
        }
        protected static void RequireOutput(IOnnxModelSource model, string name, ModelValidationResult result)
        {
            if (model == null || !model.HasInspectedMetadata) return;
            if (!model.Outputs.Any(x => x.Name == name)) result.Error("missing_output", $"Model '{model.SourceName}' is missing output '{name}'.");
        }
        protected static void RequireSchema(IOnnxModelSource model, ModelValidationResult result)
        {
            if (model == null || !model.HasInspectedMetadata) return;
            if (model.Inputs.Count == 0) result.Error("missing_inputs", $"Model '{model.SourceName}' has no inspected inputs.");
            if (model.Outputs.Count == 0) result.Error("missing_outputs", $"Model '{model.SourceName}' has no inspected outputs.");
        }
    }
}
