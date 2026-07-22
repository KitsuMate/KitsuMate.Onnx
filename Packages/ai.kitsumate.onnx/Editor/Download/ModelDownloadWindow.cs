using System;
using System.Collections.Generic;
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
        private string destination;
        private string expectedFamily;
        private string token;
        private string[] variants = Array.Empty<string>();
        private int selectedVariant;
        private string message;
        private float progress;
        private bool busy;
        private CancellationTokenSource cancellation;
        private bool HasFixedVariant => !string.IsNullOrWhiteSpace(initialRequest?.Variant);

        public static ModelDownloadWindow Show(ModelDownloadRequest request, Action<ModelDownloadResult> onCompleted)
        {
            if (request == null) throw new ArgumentNullException(nameof(request));
            var window = GetWindow<ModelDownloadWindow>(true, "Download Models", true);
            window.minSize = new Vector2(460, 260);
            window.initialRequest = request;
            window.completed = onCompleted;
            window.repository = request.Repository;
            window.revision = request.Revision;
            window.destination = request.Destination;
            window.expectedFamily = request.ExpectedFamily;
            window.token = TokenStorage.LoadToken("huggingface.co") ?? string.Empty;
            window.Show();
            _ = window.LoadVariantsAsync();
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
                destination = EditorGUILayout.TextField("Destination", destination);
                token = EditorGUILayout.PasswordField("Access Token", token);
                if (EditorGUI.EndChangeCheck()) variants = Array.Empty<string>();

                EditorGUILayout.Space();
                if (variants.Length == 0)
                {
                    if (GUILayout.Button("Load Variants")) _ = LoadVariantsAsync();
                }
                else
                {
                    using (new EditorGUI.DisabledScope(HasFixedVariant))
                        selectedVariant = EditorGUILayout.Popup("Variant", Mathf.Clamp(selectedVariant, 0, variants.Length - 1), variants);
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

            EditorGUILayout.Space();
            EditorGUILayout.HelpBox(
                "Variants are detected from the repository's onnx/ folder. Downloads are verified before the selected ModelSet is changed.",
                MessageType.None);
        }

        private async Awaitable LoadVariantsAsync()
        {
            if (!Begin("Scanning repository...")) return;
            try
            {
                ModelDownloadRequest request = Request(string.Empty);
                IReadOnlyList<string> names = await ModelDownloader.GetVariantsAsync(request, token, cancellation.Token);
                variants = names is string[] array ? array : new List<string>(names).ToArray();
                if (variants.Length == 0) throw new InvalidOperationException("The repository contains no supported variants.");
                int requested = Array.FindIndex(variants, name => string.Equals(name, initialRequest?.Variant, StringComparison.OrdinalIgnoreCase));
                if (HasFixedVariant && requested < 0)
                    throw new InvalidOperationException($"Model variant '{initialRequest.Variant}' was not found.");
                selectedVariant = requested >= 0 ? requested : 0;
                SaveToken();
                message = $"Found {variants.Length} model variant{(variants.Length == 1 ? string.Empty : "s")}.";
            }
            catch (OperationCanceledException) { message = "Cancelled."; }
            catch (Exception exception) { ShowError(exception); }
            finally { End(); }
        }

        private async Awaitable DownloadAsync()
        {
            if (!Begin("Downloading models...")) return;
            try
            {
                string variant = variants[Mathf.Clamp(selectedVariant, 0, variants.Length - 1)];
                var reporter = new Progress<float>(value => { progress = value; Repaint(); });
                ModelDownloadResult result = await ModelDownloader.DownloadAsync(Request(variant), token, reporter, cancellation.Token);
                completed?.Invoke(result);
                SaveToken();
                message = $"Installed {result.Identity.ModelId} ({result.Identity.Variant}).";
            }
            catch (OperationCanceledException) { message = "Cancelled."; }
            catch (Exception exception) { ShowError(exception); }
            finally { End(); }
        }

        private ModelDownloadRequest Request(string variant)
        {
            return new ModelDownloadRequest(repository, revision, variant, destination, expectedFamily);
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
    }
}
