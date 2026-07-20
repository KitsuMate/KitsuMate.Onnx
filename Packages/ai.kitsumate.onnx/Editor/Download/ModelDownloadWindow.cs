using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using Unity.EditorCoroutines.Editor;

namespace KitsuMate.Onnx.Editor.Download
{
    /// <summary>
    /// Editor window for downloading models from git repositories.
    /// Opened programmatically with a ModelDownloadRequest configuration.
    /// </summary>
    public class ModelDownloadWindow : EditorWindow
    {
        private ModelDownloadRequest _request;
        private int _selectedRepoIndex;
        private int _selectedVariantIndex;
        private string _customRepoInput = "";
        private bool _useCustomRepo;
        private string _token = "";
        private bool _replaceExisting = true;
        
        // State
        private bool _isFetchingFiles;
        private bool _isDownloading;
        private List<RepoFile> _currentFiles;
        private List<int> _availableVariantIndices;
        private string _currentDomain;
        private string _treeApiUrl;
        private string _rawBaseUrl;
        private string _errorMessage;
        private string _statusMessage;
        
        // Download progress
        private int _currentFileIndex;
        private int _totalFiles;
        private float _currentFileProgress;
        private string _currentFileName;
        private Dictionary<string, string> _downloadedFiles;
        
        // Coroutines
        private EditorCoroutine _fetchCoroutine;
        private EditorCoroutine _downloadCoroutine;
        
        // Download summary (computed when variant selection changes)
        private int _resolvedFileCount;
        private long _resolvedTotalSize;
        
        // Animation
        private double _lastRepaintTime;
        private int _spinnerFrame;
        private static readonly string[] SpinnerFrames = { "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏" };
        
        /// <summary>
        /// Opens the download window with the specified configuration.
        /// </summary>
        public static ModelDownloadWindow Show(ModelDownloadRequest request)
        {
            if (request == null)
                throw new ArgumentNullException(nameof(request));
            
            var window = GetWindow<ModelDownloadWindow>(true, request.WindowTitle ?? "Download Models", true);
            window._request = request;
            window._replaceExisting = request.ReplaceExisting;
            window.minSize = new Vector2(500, 400);
            window.Initialize();
            window.Show();
            
            return window;
        }
        
        private void Initialize()
        {
            _selectedRepoIndex = 0;
            _selectedVariantIndex = -1;
            _availableVariantIndices = new List<int>();
            _currentFiles = null;
            _errorMessage = null;
            _statusMessage = null;
            _downloadedFiles = new Dictionary<string, string>();
            
            // Start fetching files for the first repo
            if (_request.Repositories != null && _request.Repositories.Length > 0)
            {
                FetchFilesForSelectedRepo();
            }
        }
        
        private void OnEnable()
        {
            EditorApplication.update += OnEditorUpdate;
        }
        
        private void OnDisable()
        {
            EditorApplication.update -= OnEditorUpdate;
            CancelOperations();
        }
        
        private void OnEditorUpdate()
        {
            // Animate spinner
            if (_isFetchingFiles || _isDownloading)
            {
                var time = EditorApplication.timeSinceStartup;
                if (time - _lastRepaintTime > 0.1)
                {
                    _lastRepaintTime = time;
                    _spinnerFrame = (_spinnerFrame + 1) % SpinnerFrames.Length;
                    Repaint();
                }
            }
        }
        
        private void OnGUI()
        {
            if (_request == null)
            {
                EditorGUILayout.HelpBox("No download request configured.", MessageType.Error);
                return;
            }
            
            EditorGUILayout.Space(10);
            
            // Lock UI during download
            EditorGUI.BeginDisabledGroup(_isDownloading);
            
            DrawRepositorySection();
            EditorGUILayout.Space(10);
            DrawVariantSection();
            EditorGUILayout.Space(10);
            DrawTokenSection();
            EditorGUILayout.Space(10);
            DrawOptionsSection();
            
            EditorGUI.EndDisabledGroup();
            
            EditorGUILayout.Space(15);
            DrawStatusSection();
            EditorGUILayout.Space(10);
            DrawButtonsSection();
        }
        
        private void DrawRepositorySection()
        {
            EditorGUILayout.LabelField("Repository", EditorStyles.boldLabel);
            
            EditorGUILayout.BeginHorizontal();
            
            // Repository dropdown
            if (_request.Repositories != null && _request.Repositories.Length > 0)
            {
                var repoNames = new string[_request.Repositories.Length];
                for (int i = 0; i < _request.Repositories.Length; i++)
                {
                    repoNames[i] = _request.Repositories[i].DisplayName;
                }
                
                EditorGUI.BeginDisabledGroup(_useCustomRepo);
                var newIndex = EditorGUILayout.Popup("Source", _selectedRepoIndex, repoNames);
                if (newIndex != _selectedRepoIndex)
                {
                    _selectedRepoIndex = newIndex;
                    FetchFilesForSelectedRepo();
                }
                EditorGUI.EndDisabledGroup();
                
                // Refresh button
                if (GUILayout.Button("↻", GUILayout.Width(25)))
                {
                    FetchFilesForSelectedRepo();
                }
            }
            
            EditorGUILayout.EndHorizontal();
            
            // Custom repo toggle and input
            EditorGUILayout.Space(5);
            _useCustomRepo = EditorGUILayout.Toggle("Use Custom Repository", _useCustomRepo);
            
            if (_useCustomRepo)
            {
                EditorGUILayout.BeginHorizontal();
                var newCustomInput = EditorGUILayout.TextField("Repository", _customRepoInput);
                if (newCustomInput != _customRepoInput)
                {
                    _customRepoInput = newCustomInput;
                }
                
                if (GUILayout.Button("Fetch", GUILayout.Width(50)))
                {
                    FetchFilesForCustomRepo();
                }
                EditorGUILayout.EndHorizontal();
                
                EditorGUILayout.HelpBox("Enter 'owner/repo' for HuggingFace or a full URL.", MessageType.Info);
            }
        }
        
        private void DrawVariantSection()
        {
            EditorGUILayout.LabelField("Variant", EditorStyles.boldLabel);
            
            if (_isFetchingFiles)
            {
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField($"{SpinnerFrames[_spinnerFrame]} Fetching available files...");
                EditorGUILayout.EndHorizontal();
            }
            else if (_currentFiles == null)
            {
                EditorGUILayout.HelpBox("Select a repository and fetch files to see available variants.", MessageType.Info);
            }
            else if (_availableVariantIndices.Count == 0)
            {
                EditorGUILayout.HelpBox(
                    "No compatible variants found in this repository.\n" +
                    "Please download files manually from the repository page.", 
                    MessageType.Warning);
                
                if (GUILayout.Button("Open Repository Page"))
                {
                    var repoInput = _useCustomRepo ? _customRepoInput : _request.Repositories[_selectedRepoIndex].RepoInput;
                    OpenRepositoryPage(repoInput);
                }
            }
            else
            {
                // Build variant names
                var variantNames = new string[_availableVariantIndices.Count];
                for (int i = 0; i < _availableVariantIndices.Count; i++)
                {
                    variantNames[i] = _request.Variants[_availableVariantIndices[i]].Name;
                }
                
                // Find current selection in available list
                var currentAvailableIndex = _availableVariantIndices.IndexOf(_selectedVariantIndex);
                if (currentAvailableIndex < 0) currentAvailableIndex = 0;
                
                var newAvailableIndex = EditorGUILayout.Popup("Quantization", currentAvailableIndex, variantNames);
                var newVariantIndex = _availableVariantIndices[newAvailableIndex];
                if (newVariantIndex != _selectedVariantIndex)
                {
                    _selectedVariantIndex = newVariantIndex;
                    UpdateResolvedFileSummary();
                }
                
                // Show download size summary
                if (_resolvedFileCount > 0)
                {
                    var sizeText = _resolvedTotalSize > 0
                        ? $"{_resolvedFileCount} files, {FormatFileSize(_resolvedTotalSize)}"
                        : $"{_resolvedFileCount} files";
                    EditorGUILayout.LabelField("Download Size", sizeText);
                }
            }
        }
        
        private void DrawTokenSection()
        {
            EditorGUILayout.LabelField("Authentication", EditorStyles.boldLabel);
            
            var domain = _currentDomain ?? "repository";
            EditorGUILayout.BeginHorizontal();
            
            EditorGUILayout.LabelField($"Token for {domain}", GUILayout.Width(200));
            _token = EditorGUILayout.PasswordField(_token);
            
            if (!string.IsNullOrEmpty(_token))
            {
                if (GUILayout.Button("×", GUILayout.Width(20)))
                {
                    _token = "";
                    if (!string.IsNullOrEmpty(_currentDomain))
                    {
                        TokenStorage.ClearToken(_currentDomain);
                    }
                }
            }
            
            EditorGUILayout.EndHorizontal();
            
            // Quick-add buttons
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Quick Setup:", GUILayout.Width(80));
            
            if (GUILayout.Button("HuggingFace", GUILayout.Width(90)))
            {
                ShowTokenPopup("huggingface.co");
            }
            if (GUILayout.Button("GitHub", GUILayout.Width(70)))
            {
                ShowTokenPopup("github.com");
            }
            if (GUILayout.Button("GitLab", GUILayout.Width(70)))
            {
                ShowTokenPopup("gitlab.com");
            }
            
            GUILayout.FlexibleSpace();
            EditorGUILayout.EndHorizontal();
        }
        
        private void DrawOptionsSection()
        {
            EditorGUILayout.LabelField("Options", EditorStyles.boldLabel);
            _replaceExisting = EditorGUILayout.Toggle("Replace Existing Files", _replaceExisting);
        }
        
        private void DrawStatusSection()
        {
            if (!string.IsNullOrEmpty(_errorMessage))
            {
                EditorGUILayout.HelpBox(_errorMessage, MessageType.Error);
            }
            else if (!string.IsNullOrEmpty(_statusMessage))
            {
                EditorGUILayout.HelpBox(_statusMessage, MessageType.Info);
            }
            
            if (_isDownloading)
            {
                EditorGUILayout.Space(5);
                
                // Overall progress
                var overallProgress = _totalFiles > 0 ? (float)_currentFileIndex / _totalFiles : 0f;
                var rect = EditorGUILayout.GetControlRect(GUILayout.Height(20));
                EditorGUI.ProgressBar(rect, overallProgress, $"Overall: {_currentFileIndex}/{_totalFiles} files");
                
                // Current file progress
                if (!string.IsNullOrEmpty(_currentFileName))
                {
                    EditorGUILayout.Space(3);
                    rect = EditorGUILayout.GetControlRect(GUILayout.Height(18));
                    EditorGUI.ProgressBar(rect, _currentFileProgress, $"{SpinnerFrames[_spinnerFrame]} {_currentFileName}: {_currentFileProgress * 100:F0}%");
                }
            }
        }
        
        private void DrawButtonsSection()
        {
            EditorGUILayout.BeginHorizontal();
            GUILayout.FlexibleSpace();
            
            if (_isDownloading)
            {
                if (GUILayout.Button("Cancel", GUILayout.Width(100), GUILayout.Height(30)))
                {
                    CancelOperations();
                }
            }
            else
            {
                EditorGUI.BeginDisabledGroup(_selectedVariantIndex < 0 || _availableVariantIndices.Count == 0 || _isFetchingFiles);
                var downloadLabel = _resolvedFileCount > 0 && _resolvedTotalSize > 0
                    ? $"Download ({FormatFileSize(_resolvedTotalSize)})"
                    : "Download";
                if (GUILayout.Button(downloadLabel, GUILayout.Width(150), GUILayout.Height(30)))
                {
                    StartDownload();
                }
                EditorGUI.EndDisabledGroup();
            }
            
            if (GUILayout.Button("Close", GUILayout.Width(80), GUILayout.Height(30)))
            {
                CancelOperations();
                Close();
            }
            
            GUILayout.FlexibleSpace();
            EditorGUILayout.EndHorizontal();
        }
        
        private void FetchFilesForSelectedRepo()
        {
            if (_request.Repositories == null || _selectedRepoIndex >= _request.Repositories.Length)
                return;
            
            var repoInput = _request.Repositories[_selectedRepoIndex].RepoInput;
            FetchFiles(repoInput);
        }
        
        private void FetchFilesForCustomRepo()
        {
            if (string.IsNullOrEmpty(_customRepoInput))
                return;
            
            FetchFiles(_customRepoInput);
        }
        
        private void FetchFiles(string repoInput)
        {
            CancelOperations();
            
            _isFetchingFiles = true;
            _currentFiles = null;
            _availableVariantIndices.Clear();
            _selectedVariantIndex = -1;
            _errorMessage = null;
            _statusMessage = null;
            
            RepoFileClient.ParseRepoInput(repoInput, out _treeApiUrl, out _rawBaseUrl, out _currentDomain);
            
            // Load stored token for this domain
            if (!string.IsNullOrEmpty(_currentDomain))
            {
                var storedToken = TokenStorage.LoadToken(_currentDomain);
                if (!string.IsNullOrEmpty(storedToken))
                {
                    _token = storedToken;
                }
            }
            
            if (string.IsNullOrEmpty(_treeApiUrl))
            {
                _isFetchingFiles = false;
                _errorMessage = "Invalid repository input.";
                return;
            }
            
            _fetchCoroutine = EditorCoroutineUtility.StartCoroutine(FetchFilesCoroutine(), this);
        }
        
        private IEnumerator FetchFilesCoroutine()
        {
            List<RepoFile> files = null;
            string error = null;
            
            yield return RepoFileClient.ListFiles(
                _treeApiUrl,
                _token,
                _currentDomain,
                result => files = result,
                err => error = err
            );
            
            _isFetchingFiles = false;
            
            if (!string.IsNullOrEmpty(error))
            {
                _errorMessage = error;
                Repaint();
                yield break;
            }
            
            _currentFiles = files ?? new List<RepoFile>();
            
            // Build variants dynamically if a builder is provided
            if (_request.VariantBuilder != null)
            {
                _request.Variants = _request.VariantBuilder(_currentFiles);
            }
            
            // Determine available variants
            _availableVariantIndices.Clear();
            if (_request.Variants != null)
            {
                for (int i = 0; i < _request.Variants.Length; i++)
                {
                    if (RepoFileClient.CanResolveAllFiles(_currentFiles, _request.Variants[i].Files))
                    {
                        _availableVariantIndices.Add(i);
                    }
                }
            }
            
            // Auto-select first available variant
            if (_availableVariantIndices.Count > 0)
            {
                _selectedVariantIndex = _availableVariantIndices[0];
                UpdateResolvedFileSummary();
            }
            
            Repaint();
        }
        
        private void UpdateResolvedFileSummary()
        {
            _resolvedFileCount = 0;
            _resolvedTotalSize = 0;
            
            if (_selectedVariantIndex < 0 || _request.Variants == null || _currentFiles == null)
                return;
            
            var variant = _request.Variants[_selectedVariantIndex];
            var resolved = RepoFileClient.ResolveFiles(_currentFiles, variant.Files);
            _resolvedFileCount = resolved.Count;
            
            foreach (var kvp in resolved)
            {
                var file = _currentFiles.Find(f => f.Path == kvp.Value);
                if (file.Size > 0)
                    _resolvedTotalSize += file.Size;
            }
        }
        
        private static string FormatFileSize(long bytes)
        {
            if (bytes < 1024) return $"{bytes} B";
            if (bytes < 1024 * 1024) return $"{bytes / 1024.0:F1} KB";
            if (bytes < 1024 * 1024 * 1024) return $"{bytes / (1024.0 * 1024):F1} MB";
            return $"{bytes / (1024.0 * 1024 * 1024):F2} GB";
        }
        
        private void StartDownload()
        {
            if (_selectedVariantIndex < 0 || _request.Variants == null)
                return;
            
            var variant = _request.Variants[_selectedVariantIndex];
            var resolvedFiles = RepoFileClient.ResolveFiles(_currentFiles, variant.Files);
            
            if (resolvedFiles.Count == 0)
            {
                _errorMessage = "No files to download.";
                return;
            }
            
            // Determine destination folder
            string destFolder;
            if (_request.TargetModelSet != null)
            {
                var assetPath = AssetDatabase.GetAssetPath(_request.TargetModelSet);
                destFolder = Path.GetDirectoryName(assetPath);
            }
            else
            {
                destFolder = "Assets";
            }
            
            // Store token if provided
            if (!string.IsNullOrEmpty(_token) && !string.IsNullOrEmpty(_currentDomain))
            {
                TokenStorage.SaveToken(_currentDomain, _token);
            }
            
            _isDownloading = true;
            _downloadedFiles.Clear();
            _errorMessage = null;
            _statusMessage = "Starting download...";
            
            _downloadCoroutine = EditorCoroutineUtility.StartCoroutine(
                DownloadFilesCoroutine(resolvedFiles, destFolder, variant.Name), this);
        }
        
        private IEnumerator DownloadFilesCoroutine(Dictionary<string, string> resolvedFiles, string destFolder, string variantName)
        {
            _totalFiles = resolvedFiles.Count;
            _currentFileIndex = 0;
            
            var fullDestFolder = Path.Combine(Application.dataPath.Replace("Assets", ""), destFolder);
            
            foreach (var kvp in resolvedFiles)
            {
                var fileKey = kvp.Key;
                var filePath = kvp.Value;
                var fileName = Path.GetFileName(filePath);
                var destPath = Path.Combine(fullDestFolder, fileName);
                var assetPath = Path.Combine(destFolder, fileName);
                
                // Check if file exists and skip if not replacing
                if (!_replaceExisting && File.Exists(destPath))
                {
                    _downloadedFiles[fileKey] = assetPath;
                    _currentFileIndex++;
                    continue;
                }
                
                _currentFileName = fileName;
                _currentFileProgress = 0f;
                _statusMessage = $"Downloading {fileName}...";
                Repaint();
                
                var url = RepoFileClient.GetRawFileUrl(_rawBaseUrl, filePath, _currentDomain);
                bool success = false;
                string error = null;
                
                yield return RepoFileClient.DownloadFile(
                    url,
                    destPath,
                    _token,
                    _currentDomain,
                    progress => { _currentFileProgress = progress; },
                    (s, e) => { success = s; error = e; }
                );
                
                if (success)
                {
                    _downloadedFiles[fileKey] = assetPath;
                }
                else
                {
                    Debug.LogWarning($"[ModelDownloadWindow] Failed to download {fileName}: {error}");
                }
                
                _currentFileIndex++;
            }
            
            // Import assets
            AssetDatabase.Refresh();
            
            // Set KitsuMate importer override for downloaded model files
            foreach (var kvp in _downloadedFiles)
            {
                var path = kvp.Value;
                if (OnnxImporterAssignment.IsFrameworkModelPath(path))
                    OnnxImporterAssignment.AssignFrameworkImporter(path);
            }
            
            // Complete
            _isDownloading = false;
            _currentFileName = null;
            
            if (_downloadedFiles.Count == _totalFiles)
            {
                _statusMessage = $"Download complete! {_downloadedFiles.Count} files downloaded.";
            }
            else
            {
                _statusMessage = $"Download finished with warnings. {_downloadedFiles.Count}/{_totalFiles} files downloaded.";
            }
            
            // Invoke callback
            _request.OnFilesDownloaded?.Invoke(_downloadedFiles);
            
            Repaint();
        }
        
        private void CancelOperations()
        {
            if (_fetchCoroutine != null)
            {
                EditorCoroutineUtility.StopCoroutine(_fetchCoroutine);
                _fetchCoroutine = null;
            }
            
            if (_downloadCoroutine != null)
            {
                EditorCoroutineUtility.StopCoroutine(_downloadCoroutine);
                _downloadCoroutine = null;
            }
            
            _isFetchingFiles = false;
            _isDownloading = false;
            
        }
        
        private void ShowTokenPopup(string domain)
        {
            var currentToken = TokenStorage.LoadToken(domain) ?? "";
            var newToken = EditorInputDialog.Show($"Token for {domain}", "Enter your Personal Access Token:", currentToken, true);
            
            if (newToken != null)
            {
                if (string.IsNullOrEmpty(newToken))
                {
                    TokenStorage.ClearToken(domain);
                }
                else
                {
                    TokenStorage.SaveToken(domain, newToken);
                }
                
                // Update current token if this is the active domain
                if (_currentDomain == domain)
                {
                    _token = newToken;
                }
            }
        }
        
        private void OpenRepositoryPage(string repoInput)
        {
            RepoFileClient.ParseRepoInput(repoInput, out _, out var rawBaseUrl, out var domain);
            
            string url;
            if (domain != null && domain.Contains("huggingface.co"))
            {
                url = rawBaseUrl?.Replace("/resolve/main/", "") ?? repoInput;
            }
            else if (domain != null && domain.Contains("github.com"))
            {
                url = rawBaseUrl?.Replace("raw.githubusercontent.com", "github.com").Replace("/main/", "") ?? repoInput;
            }
            else
            {
                url = repoInput.StartsWith("http") ? repoInput : $"https://huggingface.co/{repoInput}";
            }
            
            Application.OpenURL(url);
        }
    }
    
    /// <summary>
    /// Simple input dialog for entering tokens.
    /// </summary>
    internal class EditorInputDialog : EditorWindow
    {
        private string _message;
        private string _value;
        private bool _isPassword;
        private bool _confirmed;
        private bool _initialized;
        
        public static string Show(string title, string message, string defaultValue = "", bool isPassword = false)
        {
            var window = CreateInstance<EditorInputDialog>();
            window.titleContent = new GUIContent(title);
            window._message = message;
            window._value = defaultValue;
            window._isPassword = isPassword;
            window.minSize = new Vector2(350, 100);
            window.maxSize = new Vector2(500, 100);
            
            window.ShowModal();
            
            return window._confirmed ? window._value : null;
        }
        
        private void OnGUI()
        {
            if (!_initialized)
            {
                _initialized = true;
                GUI.FocusControl("InputField");
            }
            
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField(_message);
            EditorGUILayout.Space(5);
            
            GUI.SetNextControlName("InputField");
            if (_isPassword)
            {
                _value = EditorGUILayout.PasswordField(_value);
            }
            else
            {
                _value = EditorGUILayout.TextField(_value);
            }
            
            EditorGUILayout.Space(10);
            EditorGUILayout.BeginHorizontal();
            GUILayout.FlexibleSpace();
            
            if (GUILayout.Button("OK", GUILayout.Width(80)))
            {
                _confirmed = true;
                Close();
            }
            
            if (GUILayout.Button("Cancel", GUILayout.Width(80)))
            {
                _confirmed = false;
                Close();
            }
            
            GUILayout.FlexibleSpace();
            EditorGUILayout.EndHorizontal();
            
            // Handle Enter key
            if (Event.current.type == EventType.KeyDown && Event.current.keyCode == KeyCode.Return)
            {
                _confirmed = true;
                Close();
            }
        }
    }
}
