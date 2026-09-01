using System;
using System.Collections.Generic;

namespace KitsuMate.Onnx.Motion
{
    public enum CharacterMotionDiagnosticSeverity { Warning, Error }

    public readonly struct CharacterMotionDiagnostic
    {
        public CharacterMotionDiagnosticSeverity Severity { get; }
        public string Code { get; }
        public string Message { get; }

        public CharacterMotionDiagnostic(CharacterMotionDiagnosticSeverity severity, string code, string message)
        {
            Severity = severity;
            Code = code;
            Message = message;
        }

        public override string ToString() => $"{Severity}: {Code}: {Message}";
    }

    public sealed class CharacterMotionValidationResult
    {
        private readonly List<CharacterMotionDiagnostic> diagnostics = new List<CharacterMotionDiagnostic>();
        public IReadOnlyList<CharacterMotionDiagnostic> Diagnostics => diagnostics;
        public bool IsValid
        {
            get
            {
                for (int i = 0; i < diagnostics.Count; i++)
                    if (diagnostics[i].Severity == CharacterMotionDiagnosticSeverity.Error) return false;
                return true;
            }
        }

        public void Error(string code, string message) => diagnostics.Add(new CharacterMotionDiagnostic(CharacterMotionDiagnosticSeverity.Error, code, message));
        public void Warning(string code, string message) => diagnostics.Add(new CharacterMotionDiagnostic(CharacterMotionDiagnosticSeverity.Warning, code, message));
        public void Clear() => diagnostics.Clear();
    }
}
