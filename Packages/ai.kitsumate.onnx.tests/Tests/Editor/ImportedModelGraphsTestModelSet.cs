namespace KitsuMate.Onnx.Tests
{
    internal sealed class ImportedModelGraphsTestModelSet : StandardModelSet
    {
        public OnnxModelReference Model = new();
        public TextFileReference Tokenizer = new();
        public override string DisplayName => "Move test";
        public override bool IsComplete => Model.IsAssigned && Tokenizer.IsAvailable;
        public override IOnnxModelSource[] GetAllModels() => new IOnnxModelSource[] { Model };
        public override TextFileReference[] GetAllTextFiles() => new[] { Tokenizer };
        public override ModelValidationResult Validate(ModelValidationContext context) => new();
    }
}
