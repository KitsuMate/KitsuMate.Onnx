using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Threading;
using UnityEditor;
using KitsuMate.Onnx.Download;
using UnityEngine;

namespace KitsuMate.Onnx.Editor.Download
{
    public sealed class ModelDownloadWindow : EditorWindow
    {
        private ModelSet modelSet;
        private Action<DownloadedModel> completed;
        private string repository;
        private string revision;
        private string expectedFamily;
        private string token;
        private bool sentis;
        private DiscoveredRepository discovered;
        private DownloadedModel cachedInstallation;
        private readonly Dictionary<string, string> selected = new(StringComparer.Ordinal);
        private string installationFolder;
        private bool advanced;
        private bool supportingFiles;
        private bool storageDetails;
        private Vector2 scroll;
        private MessageType messageType = MessageType.Info;
        private readonly Dictionary<string, string> supplied = new(StringComparer.Ordinal);
        private IReadOnlyDictionary<string, string> available = new Dictionary<string, string>();
        private DiscoveredFile[] required = Array.Empty<DiscoveredFile>();
        private string message;
        private float progress;
        private bool busy;
        private CancellationTokenSource cancellation;

        public static ModelDownloadWindow Show(ModelSet set, Action<DownloadedModel> onCompleted = null, bool useSentis = false)
        {
            if (set == null) throw new ArgumentNullException(nameof(set));
            string assetPath = AssetDatabase.GetAssetPath(set);
            string assetGuid = AssetDatabase.AssetPathToGUID(assetPath);
            if (string.IsNullOrEmpty(assetGuid))
                throw new InvalidOperationException("Save the model set before downloading files.");
            if (string.IsNullOrEmpty(set.Download.installationOwnerGuid))
            {
                if (string.IsNullOrWhiteSpace(set.Download.installationFolder))
                    set.Download.installationFolder = "model-sets/" + assetGuid;
                set.Download.installationOwnerGuid = assetGuid;
                EditorUtility.SetDirty(set);
                AssetDatabase.SaveAssetIfDirty(set);
            }
            else if (set.Download.installationOwnerGuid != assetGuid)
            {
                set.Download.installationOwnerGuid = assetGuid;
                set.Download.installationFolder = "model-sets/" + assetGuid;
                EditorUtility.SetDirty(set);
                AssetDatabase.SaveAssetIfDirty(set);
            }
            var window = CreateInstance<ModelDownloadWindow>();
            window.titleContent = new GUIContent("Download Models");
            window.minSize = new Vector2(560, 380);
            window.installationFolder = set.Download.installationFolder;
            window.modelSet = set;
            window.completed = onCompleted;
            window.repository = set.Download.repository;
            window.revision = set.Download.revision;
            window.expectedFamily = set.Download.family;
            bool useDefault = string.IsNullOrWhiteSpace(window.repository);
            if (useDefault)
            {
                var suggestion = set.RepositorySuggestions.FirstOrDefault();
                window.repository = suggestion.Repository;
                window.expectedFamily = suggestion.Family;
                window.revision = "";
            }
            window.sentis = useSentis;
            foreach (var item in set.Download.artifacts) window.selected[item.role] = item.path;
            window.token = TokenStorage.LoadToken("huggingface.co") ?? string.Empty;
            if (string.IsNullOrWhiteSpace(window.installationFolder)) window.installationFolder = "model-sets/" + assetGuid;
            window.Show();
            try
            {
                var installed = window.Store().Read();
                if (installed != null && installed.Identity.ModelId == window.repository?.Split('/').Last() &&
                    (string.IsNullOrEmpty(window.expectedFamily) || installed.Identity.Family == window.expectedFamily) &&
                    (string.IsNullOrEmpty(installed.Repository) || installed.Repository == window.repository) &&
                    (string.IsNullOrEmpty(window.revision) || window.revision == installed.Identity.Revision) &&
                    window.selected.All(choice => installed.Files.Any(file => file.Role == choice.Key && file.Path == choice.Value)))
                {
                    window.cachedInstallation = installed;
                    window.required = installed.Files.ToArray();
                    window.available = installed.Files.ToDictionary(file => file.Path, file => Path.Combine(installed.DirectoryPath, file.Path));
                    window.revision = installed.Identity.Revision;
                    foreach (var file in installed.Files.Where(file => file.Path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))) window.selected[file.Role] = file.Path;
                }
            }
            catch (Exception exception) { window.message = exception.Message; }
            if (!string.IsNullOrWhiteSpace(window.repository)) _ = window.ScanAsync();
            return window;
        }

        private void SaveSelection()
        {
            if (modelSet == null) return;
            Undo.RecordObject(modelSet, "Configure model download");
            modelSet.Download.repository = repository;
            modelSet.Download.revision = revision;
            modelSet.Download.family = expectedFamily;
            modelSet.Download.graphRoles = modelSet.DownloadGraphRoles?.ToArray() ?? Array.Empty<ModelGraphRole>();
            modelSet.Download.requiredCompanionRoles = modelSet.DownloadRequiredCompanionRoles;
            modelSet.Download.installationFolder = installationFolder;
            modelSet.Download.artifacts = (discovered != null ? SelectedArtifacts() : selected).Select(pair => new ModelDownloadProfile.ArtifactSelection { role = pair.Key, path = pair.Value }).ToArray();
            EditorUtility.SetDirty(modelSet);
            AssetDatabase.SaveAssetIfDirty(modelSet);
        }

        private void OnDisable()
        {
            cancellation?.Cancel();
        }

        private void OnGUI()
        {
            if (modelSet == null) { Close(); return; }
            scroll = EditorGUILayout.BeginScrollView(scroll);
            EditorGUILayout.LabelField(modelSet.DisplayName, EditorStyles.boldLabel);
            using (new EditorGUI.DisabledScope(busy))
            {
                string previousRepository = repository;
                string previousRevision = revision;
                DrawRepositoryField();
                advanced = EditorGUILayout.Foldout(advanced, "Repository and authentication", true);
                if (advanced)
                {
                    DrawRevisionField();
                    token = EditorGUILayout.PasswordField("Access token", token);
                    EditorGUILayout.LabelField("Refresh the repository after changing the access token.", EditorStyles.miniLabel);
                }
                if (previousRepository != repository || previousRevision != revision)
                {
                    discovered = null;
                    cachedInstallation = null;
                    selected.Clear();
                    supplied.Clear();
                    required = Array.Empty<DiscoveredFile>();
                    if (previousRepository != repository)
                    {
                        expectedFamily = modelSet.RepositorySuggestions.FirstOrDefault(item => item.Repository == repository).Family;
                    }
                    if (!string.IsNullOrWhiteSpace(repository)) _ = ScanAsync();
                }
                EditorGUILayout.Space(8);
                if (discovered != null)
                {
                    DrawArtifacts();
                    supportingFiles = EditorGUILayout.Foldout(supportingFiles,
                        $"Supporting files ({discovered.CommonFiles.Length})", true);
                    if (supportingFiles)
                        foreach (var file in discovered.CommonFiles) DrawFile(file);
                    using (new EditorGUILayout.HorizontalScope())
                    {
                        GUILayout.FlexibleSpace();
                        if (GUILayout.Button("Use files from folder…", EditorStyles.miniButton, GUILayout.Width(165))) SupplyFolder();
                    }
                }
                else if (cachedInstallation != null)
                {
                    foreach (var file in cachedInstallation.Files.Where(file => file.Path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase)))
                        DrawFile(file);
                    EditorGUILayout.LabelField("Available offline. Refresh the repository to change variants.", EditorStyles.wordWrappedMiniLabel);
                }
                EditorGUILayout.Space(8);
                if (EditorUserBuildSettings.activeBuildTarget == BuildTarget.WebGL)
                    EditorGUILayout.HelpBox(sentis
                        ? "WebGL uses imported Unity AI Inference assets. Model compatibility must still be tested in a player."
                        : "This raw ONNX setup is not supported on WebGL. Use a Unity AI Inference model set with imported assets. Browser downloads are not supported.",
                        sentis ? MessageType.Info : MessageType.Warning);
                DrawInterruptedUpdateRecovery();
                storageDetails = EditorGUILayout.Foldout(storageDetails, "Storage details", true);
                if (storageDetails)
                {
                    EditorGUILayout.LabelField("Download root", "Application.persistentDataPath");
                    EditorGUILayout.SelectableLabel(Destination(), EditorStyles.wordWrappedMiniLabel, GUILayout.Height(34));
                    using (new EditorGUILayout.HorizontalScope())
                    {
                        if (GUILayout.Button("Copy path", EditorStyles.miniButton)) EditorGUIUtility.systemCopyBuffer = Destination();
                        if (GUILayout.Button("Open in Explorer", EditorStyles.miniButton)) EditorUtility.RevealInFinder(Destination());
                    }
                    EditorGUILayout.LabelField("Completed files are kept after cancellation. The interrupted file restarts on retry. Downloads are verified against the repository SHA-256 hash before installation.", EditorStyles.wordWrappedMiniLabel);
                }
            }
            if (!string.IsNullOrWhiteSpace(message) && !busy) EditorGUILayout.HelpBox(message, messageType);
            EditorGUILayout.EndScrollView();
            EditorGUILayout.Space(6);
            if (busy)
            {
                Rect rect = EditorGUILayout.GetControlRect(false, EditorGUIUtility.singleLineHeight);
                EditorGUI.ProgressBar(rect, progress, message ?? "Working…");
                using (new EditorGUI.DisabledScope(cancellation == null || cancellation.IsCancellationRequested))
                    if (GUILayout.Button("Cancel")) cancellation.Cancel();
            }
            else
            {
                int missing = required.Count(file => !available.ContainsKey(file.Path));
                long bytes = required.Where(file => !available.ContainsKey(file.Path)).Sum(file => Math.Max(0, file.Size));
                if (discovered != null || cachedInstallation != null) EditorGUILayout.LabelField($"{required.Length - missing} of {required.Length} files available · {EditorUtility.FormatBytes(bytes)} to download");
                using (new EditorGUI.DisabledScope((discovered == null && cachedInstallation == null) || required.Length == 0))
                    if (GUILayout.Button(missing > 0 ? "Download all missing files" : "Assign downloaded files", GUILayout.Height(26))) _ = DownloadAsync();
            }
        }

        private ModelInstallationStore Store() => new(
            (OnnxSettings.Load() ?? throw new InvalidOperationException("Configure OnnxSettings before setting up models.")).InstallationRoot,
            installationFolder);

        private void DrawInterruptedUpdateRecovery()
        {
            ModelInstallationStore store;
            try { store = Store(); }
            catch { return; }
            if (!store.HasPendingBinding) return;
            EditorGUILayout.HelpBox(
                "A model update was interrupted after new files were installed. Restore the previous files if model assignment did not finish, or keep the current files if assignment completed.",
                MessageType.Warning);
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Restore previous files")) RecoverInterruptedUpdate(store, false);
                if (GUILayout.Button("Keep current files")) RecoverInterruptedUpdate(store, true);
            }
        }

        private void RecoverInterruptedUpdate(ModelInstallationStore store, bool keepCurrent)
        {
            try
            {
                EnsureNoActiveRuntime();
                if (keepCurrent) store.CompleteBinding();
                else store.RestorePrevious();
                cachedInstallation = store.Read();
                message = keepCurrent ? "Kept the current installation." : "Restored the previous installation.";
                messageType = MessageType.Info;
                if (discovered != null) RefreshAvailability();
                Repaint();
            }
            catch (Exception exception)
            {
                message = exception.Message;
                messageType = MessageType.Error;
            }
        }

        private void RefreshAvailability()
        {
            if (discovered == null) return;
            required = ModelInstallationStore.RequiredFiles(discovered, SelectedArtifacts());
            available = Store().AvailableFiles(discovered, required, supplied);
        }

        private void DrawFile(DiscoveredFile file, string role = null, DiscoveredArtifact[] choices = null)
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                EditorGUILayout.LabelField(new GUIContent(role == null ? Path.GetFileName(file.Path) : Humanize(role), file.Path), GUILayout.Width(135));
                if (choices != null)
                {
                    int current = Array.FindIndex(choices, choice => choice.Model.Path == selected[role]);
                    int next = EditorGUILayout.Popup(Math.Max(0, current), choices.Select(choice => new GUIContent(choice.Type.ToUpperInvariant(), choice.Model.Path)).ToArray(), GUILayout.MinWidth(65));
                    if (next != current)
                    {
                        selected[role] = choices[next].Model.Path;
                        if (role == "decoder") SelectCachedDecoder(choices[next]);
                        RefreshAvailability();
                        file = choices[next].Model;
                    }
                }
                else GUILayout.FlexibleSpace();
                bool exists = available.ContainsKey(file.Path);
                GUILayout.Label(supplied.ContainsKey(file.Path) ? "Local file" : exists ? "Downloaded" : "Missing", GUILayout.Width(80));
                using (new EditorGUI.DisabledScope(discovered == null))
                    if (GUILayout.Button(exists ? "Redownload" : "Download", EditorStyles.miniButton, GUILayout.Width(85))) _ = DownloadFileAsync(file);
                if (GUILayout.Button("Choose file...", EditorStyles.miniButton, GUILayout.Width(95))) SupplyFile(file);
                if (supplied.ContainsKey(file.Path) && GUILayout.Button("Reset", EditorStyles.miniButton, GUILayout.Width(45)))
                { supplied.Remove(file.Path); RefreshAvailability(); Repaint(); }
            }
        }

        private async Awaitable DownloadFileAsync(DiscoveredFile file)
        {
            if (!Begin("Downloading " + Path.GetFileName(file.Path))) return;
            try
            {
                var reporter = new Progress<float>(value => { progress = value; Repaint(); });
                await Store().DownloadFileAsync(discovered, file, token, reporter, cancellation.Token);
                supplied.Remove(file.Path);
                message = "Downloaded " + Path.GetFileName(file.Path) + ". Assign downloaded files when the selection is complete.";
            }
            catch (OperationCanceledException) { message = "Cancelled. The previous file is unchanged."; }
            catch (Exception exception) { ShowError(exception); }
            finally { RefreshAvailability(); End(); }
        }

        private void SupplyFile(DiscoveredFile file)
        {
            string path = EditorUtility.OpenFilePanel("Select " + file.Path, "", Path.GetExtension(file.Path).TrimStart('.'));
            if (string.IsNullOrEmpty(path)) return;
            if (file.Size <= 0 || new FileInfo(path).Length != file.Size)
            {
                message = "The selected file size does not match this repository revision and variant.";
                messageType = MessageType.Error;
                return;
            }
            supplied[file.Path] = path;
            // External weights normally sit beside their graph. Reuse matching companions too.
            foreach (var candidate in required.Where(candidate => Path.GetDirectoryName(candidate.Path) == Path.GetDirectoryName(file.Path)))
            {
                string neighbor = Path.Combine(Path.GetDirectoryName(path), Path.GetFileName(candidate.Path));
                if (candidate.Size > 0 && File.Exists(neighbor) && new FileInfo(neighbor).Length == candidate.Size) supplied[candidate.Path] = neighbor;
            }
            message = null;
            RefreshAvailability();
            Repaint();
        }

        private void SupplyFolder()
        {
            string folder = EditorUtility.OpenFolderPanel("Select model folder (repository layout)", "", "");
            if (string.IsNullOrEmpty(folder)) return;
            int count = 0;
            foreach (var file in required)
            {
                string path = Path.Combine(folder, file.Path);
                if (file.Size > 0 && File.Exists(path) && new FileInfo(path).Length == file.Size) { supplied[file.Path] = path; count++; }
            }
            RefreshAvailability();
            message = $"Found {count} matching files. Missing files will be downloaded.";
            messageType = MessageType.Info;
        }

        private void DrawRepositoryField()
        {
            Rect field = EditorGUILayout.GetControlRect();
            field = EditorGUI.PrefixLabel(field, new GUIContent("Hugging Face"));
            var refresh = new Rect(field.xMax - 22, field.y, 22, field.height);
            field.width -= 24;
            using (new EditorGUI.DisabledScope(string.IsNullOrWhiteSpace(repository)))
                if (GUI.Button(refresh, new GUIContent("\u21bb", "Refresh repository"), EditorStyles.miniButton)) _ = ScanAsync();
            var arrow = new Rect(field.xMax - 22, field.y, 22, field.height);
            field.width -= 22;
            string next = EditorGUI.DelayedTextField(field, repository ?? string.Empty);
            if (next != repository)
            {
                repository = next;
                revision = "";
            }
            if (!EditorGUI.DropdownButton(arrow, new GUIContent("", "Choose a supported repository or type your own"), FocusType.Keyboard)) return;
            var menu = new GenericMenu();
            foreach (var preset in modelSet.RepositorySuggestions)
            {
                var choice = preset;
                menu.AddItem(new GUIContent(choice.Repository.Replace("/", "\u2215")),
                    string.Equals(repository, choice.Repository, StringComparison.OrdinalIgnoreCase), () =>
                    {
                        repository = choice.Repository;
                        cachedInstallation = null;
                        expectedFamily = choice.Family;
                        supplied.Clear();
                        revision = "";
                        discovered = null;
                        selected.Clear();
                        message = null;
                        GUI.FocusControl(null);
                        Repaint();
                        _ = ScanAsync();
                    });
            }
            menu.DropDown(arrow);
        }

        private void DrawRevisionField()
        {
            Rect field = EditorGUILayout.GetControlRect();
            field = EditorGUI.PrefixLabel(field, new GUIContent("Branch / revision", "Leave empty for the default branch, or enter a branch, tag, or commit."));
            var arrow = new Rect(field.xMax - 22, field.y, 22, field.height);
            field.width -= 22;
            revision = EditorGUI.DelayedTextField(field, revision ?? string.Empty);
            using (new EditorGUI.DisabledScope(string.IsNullOrWhiteSpace(repository)))
                if (EditorGUI.DropdownButton(arrow, new GUIContent("", "Choose a repository branch"), FocusType.Keyboard))
                    _ = ShowBranchesAsync(arrow);
        }

        private async Awaitable ShowBranchesAsync(Rect position)
        {
            if (!Begin("Loading branches...")) return;
            try
            {
                string[] branches = await HuggingFaceModelRepository.GetBranchesAsync(repository, token, cancellation.Token);
                cancellation.Token.ThrowIfCancellationRequested();
                var menu = new GenericMenu();
                void Choose(string value)
                {
                    revision = value;
                    supplied.Clear();
                    cachedInstallation = null;
                    discovered = null;
                    selected.Clear();
                    message = null;
                    GUI.FocusControl(null);
                    Repaint();
                    EditorApplication.delayCall += () => { if (this != null) _ = ScanAsync(); };
                }
                foreach (string branch in branches)
                {
                    string value = branch;
                    menu.AddItem(new GUIContent(value.Replace("/", "\u2215")), revision == value, () => Choose(value));
                }
                menu.DropDown(position);
                message = null;
            }
            catch (OperationCanceledException) { message = "Cancelled."; }
            catch (Exception exception) { message = "Could not load branches: " + exception.Message; }
            finally { End(); }
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

                var artifact = choices.First(choice => choice.Model.Path == selected[role]);
                DrawFile(artifact.Model, role, choices);
                artifact = choices.First(choice => choice.Model.Path == selected[role]);
                foreach (var file in artifact.Files.Where(file => file.Path != artifact.Model.Path)) DrawFile(file);

            }
        }

        private IEnumerable<string> OrderedRoles()
        {
            foreach (string role in discovered.RequiredRoles)
                yield return role;
            if (ShowCachedDecoder())
                yield return "decoder-with-past";

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
                ? artifacts.Where(artifact => HuggingFaceModelRepository.IsSentisArtifact(artifact.Type)).ToArray()
                : artifacts;
        }

        private async Awaitable ScanAsync()
        {
            if (!Begin("Scanning repository...")) return;
            try
            {
                var catalog = await HuggingFaceModelRepository.ScanAsync(Request(), token, cancellation.Token);
                cancellation.Token.ThrowIfCancellationRequested();
                if (modelSet == null) throw new InvalidOperationException("The selected model set no longer exists.");
                var roles = modelSet.DownloadCompanionRoles;
                bool hasTokenizer = catalog.CommonFiles.Any(file => file.Role == "tokenizer") && roles.Contains("tokenizer");
                discovered = new DiscoveredRepository(catalog.Owner, catalog.Name, catalog.Family, catalog.Revision,
                    catalog.Artifacts, catalog.CommonFiles.Where(file => roles.Contains(file.Role) &&
                        !(hasTokenizer && file.Role == "vocabulary")).ToArray(), catalog.RequiredRoles);
                foreach (string role in discovered.Artifacts.Keys)
                {
                    DiscoveredArtifact[] choices = Choices(role);
                    if (choices.Length > 0 && (!selected.TryGetValue(role, out string saved) || !choices.Any(choice => choice.Model.Path == saved))) selected[role] = choices[0].Model.Path;
                }
                EnsureRequiredChoices();
                SaveToken();
                RefreshAvailability();
                message = null;
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
                EnsureNoActiveRuntime();
                if (discovered == null && cachedInstallation != null)
                {
                    var current = Store().Read() ?? throw new InvalidOperationException("Installation files have changed. Refresh the repository to repair the installation.");
                    if (supplied.Count > 0) throw new InvalidOperationException("Refresh the repository before applying replacement files.");
                    await ApplyAsync(current);
                    return;
                }
                EnsureRequiredChoices();
                var reporter = new Progress<float>(value => { progress = value; Repaint(); });
                var store = Store();
                RefreshAvailability();
                var installed = store.Read();
                bool matches = installed != null && installed.Identity.Revision == discovered.Revision &&
                    installed.Identity.Family == discovered.Family && installed.Identity.ModelId == discovered.Name &&
                    (string.IsNullOrEmpty(installed.Repository) || installed.Repository == discovered.Repository) &&
                    required.All(file => installed.Files.Any(existing => existing.Path == file.Path &&
                        existing.Size == file.Size && existing.Sha256 == file.Sha256) &&
                        available.TryGetValue(file.Path, out string source) && Path.GetFullPath(source) == Path.GetFullPath(Path.Combine(installed.DirectoryPath, file.Path))) && supplied.Count == 0;
                var fileReporter = new Progress<(string Path, long Bytes, long Total)>(value =>
                {
                    message = $"{Path.GetFileName(value.Path)} · {EditorUtility.FormatBytes(value.Bytes)} / {EditorUtility.FormatBytes(value.Total)}";
                    Repaint();
                });
                bool installedNew = !matches;
                if (installedNew) await store.InstallAsync(discovered, SelectedArtifacts(), reporter, cancellation.Token, token, supplied,
                    (path, bytes, total) => ((IProgress<(string Path, long Bytes, long Total)>)fileReporter).Report((path, bytes, total)),
                    retainPreviousUntilBinding: true);
                try
                {
                    EnsureNoActiveRuntime();
                    DownloadedModel result = store.Read() ?? throw new InvalidOperationException("Installation is incomplete.");
                    await ApplyAsync(result);
                }
                catch
                {
                    if (installedNew) store.RestorePrevious();
                    throw;
                }
                if (installedNew)
                {
                    try { store.CompleteBinding(); }
                    catch (IOException exception)
                    {
                        message = "Model assignment succeeded, but old files could not be removed: " + exception.Message;
                        messageType = MessageType.Warning;
                    }
                }
                supplied.Clear();
                RefreshAvailability();
                SaveToken();
            }
            catch (OperationCanceledException) { message = "Cancelled. Completed files are kept for the next attempt."; }
            catch (Exception exception) { ShowError(exception); }
            finally { try { RefreshAvailability(); } finally { End(); } }
        }

        private void EnsureNoActiveRuntime()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
                throw new InvalidOperationException("Stop Play Mode before changing installed model files.");
            if (InferenceEngineRuntimeBase.LiveRuntimes.Any(runtime => runtime.SourceModelSet == modelSet))
                throw new InvalidOperationException("Unload previews and other runtimes using this model set before changing its files.");
            if (!sentis && (modelSet.GetAllModels().OfType<OnnxModelReference>().Any(source =>
                    source.Kind == OnnxModelReference.SourceKind.Asset && source.Asset != null) ||
                modelSet.GetAllTextFiles().Any(source => source != null &&
                    source.Kind == OnnxModelReference.SourceKind.Asset && source.Asset != null)))
                throw new InvalidOperationException(
                    "This model set uses imported Assets. Move its files into the data folder before applying a new download; the download window cannot replace imported files in place yet.");
        }

        private async Awaitable ApplyAsync(DownloadedModel result)
        {
            // Applying is explicit and only happens after all required files are available.
            string before = EditorJsonUtility.ToJson(modelSet);
            try
            {
                revision = result.Identity.Revision;
                SaveSelection();
                if (modelSet is StandardModelSet standard) standard.SetDownloadMetadata(result.Identity, Array.Empty<string>());
                completed?.Invoke(result);
                await modelSet.ApplyInstallationInEditorAsync(result, cancellation.Token);
                AssetDatabase.SaveAssetIfDirty(modelSet);
            }
            catch
            {
                EditorJsonUtility.FromJsonOverwrite(before, modelSet);
                EditorUtility.SetDirty(modelSet);
                AssetDatabase.SaveAssetIfDirty(modelSet);
                throw;
            }
            message = "Download complete. File references assigned to the model set.";
        }

        private string Destination()
        {
            try { return Store().DirectoryPath; }
            catch { return "Configure OnnxSettings to choose the download root."; }
        }

        private IReadOnlyDictionary<string, string> SelectedArtifacts()
        {
            var result = selected.Where(pair => OrderedRoles().Contains(pair.Key)).ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
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
            return new ModelDownloadRequest(repository, revision, expectedFamily,
                modelSet.DownloadGraphRoles, modelSet.DownloadRequiredCompanionRoles);
        }

        private bool Begin(string status)
        {
            if (busy) return false;
            cancellation?.Dispose();
            cancellation = new CancellationTokenSource();
            busy = true;
            messageType = MessageType.Info;
            progress = 0f;
            message = status;
            Repaint();
            return true;
        }

        private void End()
        {
            busy = false;
            cancellation?.Dispose();
            cancellation = null;
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
            messageType = MessageType.Error;
            Debug.LogException(exception);
        }

        private static string Humanize(string role)
        {
            return string.Join(" ", role.Split('-')
                .Select(part => char.ToUpperInvariant(part[0]) + part.Substring(1)));
        }
    }
}
