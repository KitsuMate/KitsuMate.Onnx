using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Editor.Download
{
    public sealed class ModelDownloadWindow : EditorWindow
    {
        private ModelDownloadRequest initialRequest;
        private Action<ModelDownloadResult> completed;
        private string repository;
        private string revision;
        private string expectedFamily;
        private string token;
        private bool sentis;
        private DiscoveredRepository discovered;
        private readonly Dictionary<string, string> selected = new(StringComparer.Ordinal);
        private string message;
        private float progress;
        private bool busy;
        private CancellationTokenSource cancellation;

        public static ModelDownloadWindow Show(ModelDownloadRequest request,
            Action<ModelDownloadResult> onCompleted)
        {
            return Show(request, onCompleted, false);
        }

        public static ModelDownloadWindow ShowForSentis(ModelDownloadRequest request,
            Action<ModelDownloadResult> onCompleted)
        {
            return Show(request, onCompleted, true);
        }

        private static ModelDownloadWindow Show(ModelDownloadRequest request,
            Action<ModelDownloadResult> onCompleted, bool useSentis)
        {
            if (request == null) throw new ArgumentNullException(nameof(request));
            var window = GetWindow<ModelDownloadWindow>(true, "Download Models", true);
            window.minSize = new Vector2(560, 340);
            window.initialRequest = request;
            window.completed = onCompleted;
            window.repository = request.Repository;
            window.revision = request.Revision;
            window.expectedFamily = request.ExpectedFamily;
            window.sentis = useSentis;
            window.token = TokenStorage.LoadToken("huggingface.co") ?? string.Empty;
            window.Show();
            _ = window.ScanAsync();
            return window;
        }

        private void OnDisable()
        {
            cancellation?.Cancel();
            cancellation?.Dispose();
            cancellation = null;
        }

        private void OnGUI()
        {
            EditorGUILayout.LabelField("Model Repository", EditorStyles.boldLabel);
            using (new EditorGUI.DisabledScope(busy))
            {
                EditorGUI.BeginChangeCheck();
                repository = EditorGUILayout.TextField("Hugging Face", repository);
                revision = EditorGUILayout.TextField("Revision", revision);
                token = EditorGUILayout.PasswordField("Access Token", token);
                if (EditorGUI.EndChangeCheck())
                {
                    discovered = null;
                    selected.Clear();
                }

                EditorGUILayout.Space();
                if (discovered == null)
                {
                    if (GUILayout.Button("Scan Repository")) _ = ScanAsync();
                }
                else
                {
                    DrawArtifacts();
                    EditorGUILayout.Space();
                    if (GUILayout.Button("Download Models")) _ = DownloadAsync();
                }
            }

            if (busy)
            {
                Rect rect = EditorGUILayout.GetControlRect(false, EditorGUIUtility.singleLineHeight);
                EditorGUI.ProgressBar(rect, progress, message ?? "Working...");
            }
            else if (!string.IsNullOrWhiteSpace(message))
            {
                EditorGUILayout.HelpBox(message, MessageType.Info);
            }

            if (sentis && HasHiddenArtifacts())
                EditorGUILayout.HelpBox(
                    "Sentis setup shows only FP32 artifacts. Other ONNX artifacts are available with ONNX Runtime.",
                    MessageType.None);
            EditorGUILayout.HelpBox(
                $"Models are installed under {ModelDownloader.ModelRoot()}. Each ONNX role can use a different artifact type.",
                MessageType.None);
        }

        private void DrawArtifacts()
        {
            EditorGUILayout.LabelField("ONNX Artifacts", EditorStyles.boldLabel);
            foreach (string role in OrderedRoles())
            {
                DiscoveredArtifact[] choices = Choices(role);
                if (choices.Length == 0)
                {
                    EditorGUILayout.HelpBox($"No compatible artifact is available for {Humanize(role)}.",
                        MessageType.Error);
                    continue;
                }

                if (!selected.TryGetValue(role, out string path) ||
                    choices.All(choice => !string.Equals(choice.Model.Path, path,
                        StringComparison.OrdinalIgnoreCase)))
                {
                    path = choices[0].Model.Path;
                    selected[role] = path;
                }

                int current = Array.FindIndex(choices, choice =>
                    string.Equals(choice.Model.Path, path, StringComparison.OrdinalIgnoreCase));
                string[] labels = choices.Select(ModelDownloader.ArtifactLabel).ToArray();
                using (new EditorGUI.DisabledScope(choices.Length == 1))
                {
                    int next = EditorGUILayout.Popup(Humanize(role), Math.Max(0, current), labels);
                    if (next != current)
                    {
                        selected[role] = choices[next].Model.Path;
                        if (string.Equals(role, "decoder", StringComparison.Ordinal))
                            SelectCachedDecoder(choices[next]);
                    }
                }
            }
        }

        private IEnumerable<string> OrderedRoles()
        {
            foreach (string role in discovered.RequiredRoles)
                yield return role;
            if (ShowCachedDecoder())
                yield return "decoder-with-past";
            foreach (string role in discovered.Artifacts.Keys.OrderBy(value => value, StringComparer.Ordinal))
                if (!discovered.RequiredRoles.Contains(role) &&
                    !string.Equals(role, "decoder-with-past", StringComparison.Ordinal))
                    yield return role;
        }

        private bool ShowCachedDecoder()
        {
            if (!discovered.Artifacts.ContainsKey("decoder-with-past")) return false;
            if (!selected.TryGetValue("decoder", out string path)) return true;
            DiscoveredArtifact decoder = discovered.Artifacts["decoder"].FirstOrDefault(choice =>
                string.Equals(choice.Model.Path, path, StringComparison.OrdinalIgnoreCase));
            return decoder == null || !decoder.IsMerged;
        }

        private void SelectCachedDecoder(DiscoveredArtifact decoder)
        {
            if (decoder.IsMerged || !discovered.Artifacts.ContainsKey("decoder-with-past")) return;
            DiscoveredArtifact[] choices = Choices("decoder-with-past");
            if (choices.Length == 0) return;
            DiscoveredArtifact match = choices.FirstOrDefault(choice =>
                    string.Equals(choice.Type, decoder.Type, StringComparison.OrdinalIgnoreCase)) ??
                choices.FirstOrDefault(choice =>
                    string.Equals(choice.Type, "default", StringComparison.OrdinalIgnoreCase)) ??
                choices[0];
            selected["decoder-with-past"] = match.Model.Path;
        }

        private DiscoveredArtifact[] Choices(string role)
        {
            if (!discovered.Artifacts.TryGetValue(role, out DiscoveredArtifact[] artifacts))
                return Array.Empty<DiscoveredArtifact>();
            return sentis
                ? artifacts.Where(artifact => ModelDownloader.IsSentisArtifact(artifact.Type)).ToArray()
                : artifacts;
        }

        private bool HasHiddenArtifacts()
        {
            return discovered != null && discovered.Artifacts.Values
                .SelectMany(artifacts => artifacts)
                .Any(artifact => !ModelDownloader.IsSentisArtifact(artifact.Type));
        }

        private async Awaitable ScanAsync()
        {
            if (!Begin("Scanning repository...")) return;
            try
            {
                discovered = await ModelDownloader.ScanAsync(Request(), token, cancellation.Token);
                revision = discovered.Revision;
                selected.Clear();
                foreach (string role in discovered.Artifacts.Keys)
                {
                    DiscoveredArtifact[] choices = Choices(role);
                    if (choices.Length > 0) selected[role] = choices[0].Model.Path;
                }
                if (selected.TryGetValue("decoder", out string decoderPath))
                    SelectCachedDecoder(discovered.Artifacts["decoder"].First(choice =>
                        string.Equals(choice.Model.Path, decoderPath, StringComparison.OrdinalIgnoreCase)));
                EnsureRequiredChoices();
                SaveToken();
                message = $"Found {selected.Count} model role{(selected.Count == 1 ? string.Empty : "s")}.";
            }
            catch (OperationCanceledException) { message = "Cancelled."; }
            catch (Exception exception) { discovered = null; ShowError(exception); }
            finally { End(); }
        }

        private async Awaitable DownloadAsync()
        {
            if (!Begin("Downloading models...")) return;
            try
            {
                EnsureRequiredChoices();
                var reporter = new Progress<float>(value => { progress = value; Repaint(); });
                ModelDownloadResult result = sentis
                    ? await ModelDownloader.DownloadForSentisAsync(Request(), SelectedArtifacts(), token, reporter,
                        cancellation.Token)
                    : await ModelDownloader.DownloadAsync(Request(), SelectedArtifacts(), token, reporter,
                        cancellation.Token);
                completed?.Invoke(result);
                SaveToken();
                message = $"Installed {result.Identity.ModelId}.";
            }
            catch (OperationCanceledException) { message = "Cancelled."; }
            catch (Exception exception) { ShowError(exception); }
            finally { End(); }
        }

        private IReadOnlyDictionary<string, string> SelectedArtifacts()
        {
            var result = new Dictionary<string, string>(selected, StringComparer.Ordinal);
            if (!ShowCachedDecoder()) result.Remove("decoder-with-past");
            return result;
        }

        private void EnsureRequiredChoices()
        {
            foreach (string role in discovered.RequiredRoles)
                if (Choices(role).Length == 0)
                    throw new InvalidOperationException(
                        $"Repository has no {Humanize(role)} artifact compatible with this engine.");
            if (ShowCachedDecoder() && Choices("decoder-with-past").Length == 0)
                throw new InvalidOperationException(
                    "The selected initial decoder requires a compatible Decoder With Past artifact.");
        }

        private ModelDownloadRequest Request()
        {
            return new ModelDownloadRequest(repository, revision, expectedFamily);
        }

        private bool Begin(string status)
        {
            if (busy) return false;
            cancellation?.Dispose();
            cancellation = new CancellationTokenSource();
            busy = true;
            progress = 0f;
            message = status;
            Repaint();
            return true;
        }

        private void End()
        {
            busy = false;
            progress = 0f;
            Repaint();
        }

        private void SaveToken()
        {
            if (string.IsNullOrWhiteSpace(token)) TokenStorage.ClearToken("huggingface.co");
            else TokenStorage.SaveToken("huggingface.co", token);
        }

        private void ShowError(Exception exception)
        {
            message = exception.Message;
            Debug.LogException(exception);
            EditorUtility.DisplayDialog("Model download failed", exception.Message, "OK");
        }

        private static string Humanize(string role)
        {
            return string.Join(" ", role.Split('-')
                .Select(part => char.ToUpperInvariant(part[0]) + part.Substring(1)));
        }
    }
}
