using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using KitsuMate.Onnx.Download;
using System.Threading;
using System.Threading.Tasks;

namespace KitsuMate.Onnx
{
    [Serializable]
    public readonly struct ModelIdentity : IEquatable<ModelIdentity>
    {
        public readonly string Family, ModelId, Revision, ContentHash;
        public ModelIdentity(string family, string modelId, string revision, string contentHash)
        { Family = family ?? ""; ModelId = modelId ?? ""; Revision = revision ?? ""; ContentHash = contentHash ?? ""; }
        public bool Equals(ModelIdentity other) => Family == other.Family && ModelId == other.ModelId && Revision == other.Revision && ContentHash == other.ContentHash;
        public override bool Equals(object obj) => obj is ModelIdentity other && Equals(other);
        public override int GetHashCode() => HashCode.Combine(Family, ModelId, Revision, ContentHash);
        public override string ToString() => $"{Family}/{ModelId}@{Revision}#{ContentHash}";
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
        [SerializeField] private ModelDownloadProfile download = new();
        public ModelDownloadProfile Download => download;
        public virtual string[] DownloadCompanionRoles => Array.Empty<string>();
        public virtual IEnumerable<(string Family, string Repository)> RepositorySuggestions =>
            Array.Empty<(string Family, string Repository)>();
        public bool UsesInstallation => !string.IsNullOrWhiteSpace(download.repository) && !GetAllModels().Any(source => source != null && source.IsAvailable);
        public ModelInstallationStore Installation(string root) => new(root, download.installationFolder);

#if UNITY_EDITOR
        public async Task ApplyInstallationInEditorAsync(DownloadedModel installation, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string assetPath = UnityEditor.AssetDatabase.GetAssetPath(this);
            if (string.IsNullOrEmpty(assetPath)) throw new InvalidOperationException("Save the model set before assigning downloaded files.");
            using var resolved = new ResolvedModelSet(this);
            resolved.Clone();
            resolved.CompanionAssetDirectory = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(assetPath),
                System.IO.Path.GetFileNameWithoutExtension(assetPath) + " Files").Replace('\\', '/');
            await resolved.Model.BindInstallationAsync(installation, resolved, cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            UnityEditor.Undo.RecordObject(this, "Assign downloaded models");
            resolved.Model.name = name;
            resolved.Model.hideFlags = hideFlags;
            UnityEditor.EditorUtility.CopySerialized(resolved.Model, this);
            UnityEditor.EditorUtility.SetDirty(this);
            UnityEditor.AssetDatabase.SaveAssetIfDirty(this);
        }
#endif

        internal async Task<ResolvedModelSet> ResolveAsync(string root, CancellationToken cancellationToken)
        {
            if (UsesInstallation && Application.platform == RuntimePlatform.WebGLPlayer)
                throw new PlatformNotSupportedException("WebGL requires imported Unity AI Inference build assets. Browser model installations are not supported.");
            var resolved = new ResolvedModelSet(this);
            resolved.Clone();
            if (!UsesInstallation) return resolved;
            try
            {
                DownloadedModel installation = Installation(root).Read()
                    ?? throw new InvalidOperationException($"{DisplayName} is not installed. Download it from the model set Inspector.");
                if (resolved.Model is StandardModelSet standard) standard.SetResolvedIdentity(installation.Identity);
                await resolved.Model.BindInstallationAsync(installation, resolved, cancellationToken);
                return resolved;
            }
            catch { resolved.Dispose(); throw; }
        }

        protected virtual Task BindInstallationAsync(DownloadedModel installation, ResolvedModelSet resources,
            CancellationToken cancellationToken) => throw new NotSupportedException($"{GetType().Name} requires imported model sources.");

        public abstract string DisplayName { get; }
        public abstract ModelIdentity Identity { get; }
        public abstract ModelCapabilities Capabilities { get; }
        public abstract bool IsComplete { get; }
        public abstract IOnnxModelSource[] GetAllModels();
        public virtual TextFileReference[] GetAllTextFiles() => Array.Empty<TextFileReference>();
        public abstract ModelValidationResult Validate(ModelValidationContext context);
    }

    public abstract class StandardModelSet : ModelSet
    {
        [Header("Model Identity")]
        [SerializeField] private string family;
        [SerializeField] private string modelId;
        [SerializeField] private string revision = "main";
        [SerializeField] private string contentHash;
        [SerializeField] private ModelCapabilities capabilities;

        [NonSerialized] private ModelIdentity? resolvedIdentity;
        internal void SetResolvedIdentity(ModelIdentity identity) => resolvedIdentity = identity;

        protected virtual string DefaultFamily => GetType().Namespace ?? "onnx";
        protected virtual string DefaultModelId => GetType().Name;
        public override ModelIdentity Identity => resolvedIdentity ?? new(string.IsNullOrWhiteSpace(family) ? DefaultFamily : family, string.IsNullOrWhiteSpace(modelId) ? DefaultModelId : modelId, revision, contentHash);
        public override ModelCapabilities Capabilities => capabilities;

#if UNITY_EDITOR
        public void SetDownloadMetadata(ModelIdentity identity, IEnumerable<string> downloadedCapabilities)
        {
            family = identity.Family;
            modelId = identity.ModelId;
            revision = identity.Revision;
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
