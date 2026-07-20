using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.Networking;

namespace KitsuMate.Onnx.Editor.Download
{
    /// <summary>
    /// Represents a file in a repository.
    /// </summary>
    public struct RepoFile
    {
        public string Path;
        public string Name;
        public long Size;
        
        public RepoFile(string path, string name, long size = 0)
        {
            Path = path;
            Name = name;
            Size = size;
        }
    }
    
    /// <summary>
    /// Client for listing and downloading files from git repositories.
    /// Supports HuggingFace, GitHub, and GitLab via their respective APIs.
    /// </summary>
    public static class RepoFileClient
    {
        /// <summary>
        /// Parses a repository input string and returns the API URLs for listing and downloading.
        /// Supports "owner/repo" format (defaults to HuggingFace) or full URLs.
        /// </summary>
        public static void ParseRepoInput(string input, out string treeApiUrl, out string rawBaseUrl, out string domain)
        {
            treeApiUrl = null;
            rawBaseUrl = null;
            domain = null;
            
            if (string.IsNullOrEmpty(input))
                return;
            
            input = input.Trim();
            
            // Handle owner/repo format -> assume HuggingFace
            if (!input.Contains("://") && Regex.IsMatch(input, @"^[\w\-\.]+/[\w\-\.]+$"))
            {
                domain = "huggingface.co";
                treeApiUrl = $"https://huggingface.co/api/models/{input}";
                rawBaseUrl = $"https://huggingface.co/{input}/resolve/main/";
                return;
            }
            
            // Parse as URL
            try
            {
                var uri = new Uri(input);
                domain = uri.Host;
                
                // HuggingFace
                if (domain.Contains("huggingface.co"))
                {
                    // Extract owner/repo from path like /owner/repo or /owner/repo/tree/main
                    var pathParts = uri.AbsolutePath.Trim('/').Split('/');
                    if (pathParts.Length >= 2)
                    {
                        var ownerRepo = $"{pathParts[0]}/{pathParts[1]}";
                        treeApiUrl = $"https://huggingface.co/api/models/{ownerRepo}";
                        rawBaseUrl = $"https://huggingface.co/{ownerRepo}/resolve/main/";
                    }
                }
                // GitHub
                else if (domain.Contains("github.com"))
                {
                    var pathParts = uri.AbsolutePath.Trim('/').Split('/');
                    if (pathParts.Length >= 2)
                    {
                        var ownerRepo = $"{pathParts[0]}/{pathParts[1]}";
                        treeApiUrl = $"https://api.github.com/repos/{ownerRepo}/contents";
                        rawBaseUrl = $"https://raw.githubusercontent.com/{ownerRepo}/main/";
                    }
                }
                // GitLab
                else if (domain.Contains("gitlab"))
                {
                    var pathParts = uri.AbsolutePath.Trim('/').Split('/');
                    if (pathParts.Length >= 2)
                    {
                        var ownerRepo = $"{pathParts[0]}/{pathParts[1]}";
                        var encodedPath = Uri.EscapeDataString(ownerRepo);
                        treeApiUrl = $"https://{domain}/api/v4/projects/{encodedPath}/repository/tree?recursive=true";
                        rawBaseUrl = $"https://{domain}/api/v4/projects/{encodedPath}/repository/files/{{path}}/raw?ref=main";
                    }
                }
                // Unknown - try to use as-is
                else
                {
                    treeApiUrl = input;
                    rawBaseUrl = input.TrimEnd('/') + "/";
                }
            }
            catch
            {
                // Invalid URL
            }
        }
        
        /// <summary>
        /// Lists files in a repository using the tree API.
        /// </summary>
        public static IEnumerator ListFiles(
            string treeApiUrl, 
            string token, 
            string domain,
            Action<List<RepoFile>> onComplete, 
            Action<string> onError)
        {
            var request = UnityWebRequest.Get(treeApiUrl);
            
            // Set auth header based on domain
            if (!string.IsNullOrEmpty(token))
            {
                if (domain != null && domain.Contains("gitlab"))
                {
                    request.SetRequestHeader("PRIVATE-TOKEN", token);
                }
                else
                {
                    request.SetRequestHeader("Authorization", $"Bearer {token}");
                }
            }
            
            request.SetRequestHeader("Accept", "application/json");
            
            yield return request.SendWebRequest();
            
            if (request.result != UnityWebRequest.Result.Success)
            {
                onError?.Invoke($"Failed to list files: {request.error}");
                request.Dispose();
                yield break;
            }
            
            var json = request.downloadHandler.text;
            request.Dispose();
            
            var files = new List<RepoFile>();
            
            // Parse based on domain
            if (domain != null && domain.Contains("huggingface.co"))
            {
                files = ParseHuggingFaceTree(json);
            }
            else if (domain != null && domain.Contains("github.com"))
            {
                files = ParseGitHubTree(json);
            }
            else if (domain != null && domain.Contains("gitlab"))
            {
                files = ParseGitLabTree(json);
            }
            else
            {
                // Try HuggingFace format first, then others
                files = ParseHuggingFaceTree(json);
                if (files.Count == 0)
                    files = ParseGitHubTree(json);
                if (files.Count == 0)
                    files = ParseGitLabTree(json);
            }
            
            onComplete?.Invoke(files);
        }
        
        /// <summary>
        /// Downloads a file from a URL to the specified path.
        /// </summary>
        public static IEnumerator DownloadFile(
            string url,
            string destPath,
            string token,
            string domain,
            Action<float> onProgress,
            Action<bool, string> onComplete)
        {
            // Ensure directory exists
            var directory = Path.GetDirectoryName(destPath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }
            
            var request = UnityWebRequest.Get(url);
            request.downloadHandler = new DownloadHandlerFile(destPath);
            
            // Set auth header
            if (!string.IsNullOrEmpty(token))
            {
                if (domain != null && domain.Contains("gitlab"))
                {
                    request.SetRequestHeader("PRIVATE-TOKEN", token);
                }
                else
                {
                    request.SetRequestHeader("Authorization", $"Bearer {token}");
                }
            }
            
            var operation = request.SendWebRequest();
            
            while (!operation.isDone)
            {
                onProgress?.Invoke(request.downloadProgress);
                yield return null;
            }
            
            if (request.result != UnityWebRequest.Result.Success)
            {
                // Clean up partial file
                if (File.Exists(destPath))
                {
                    try { File.Delete(destPath); } catch { }
                }
                
                onComplete?.Invoke(false, $"Download failed: {request.error}");
                request.Dispose();
                yield break;
            }
            
            onProgress?.Invoke(1f);
            onComplete?.Invoke(true, null);
            request.Dispose();
        }
        
        /// <summary>
        /// Builds the raw download URL for a file.
        /// </summary>
        public static string GetRawFileUrl(string rawBaseUrl, string filePath, string domain)
        {
            if (string.IsNullOrEmpty(rawBaseUrl) || string.IsNullOrEmpty(filePath))
                return null;
            
            // GitLab uses a different pattern with {path} placeholder
            if (rawBaseUrl.Contains("{path}"))
            {
                return rawBaseUrl.Replace("{path}", Uri.EscapeDataString(filePath));
            }
            
            return rawBaseUrl + filePath;
        }
        
        /// <summary>
        /// Checks if a file path matches a wildcard pattern.
        /// Supports * as wildcard and path prefixes like "onnx/*.onnx".
        /// </summary>
        public static bool MatchesPattern(string filePath, string pattern)
        {
            if (string.IsNullOrEmpty(filePath) || string.IsNullOrEmpty(pattern))
                return false;
            
            // Normalize path separators
            filePath = filePath.Replace('\\', '/');
            pattern = pattern.Replace('\\', '/');
            
            // Convert wildcard pattern to regex
            var regexPattern = "^" + Regex.Escape(pattern)
                .Replace("\\*", ".*")
                .Replace("\\?", ".") + "$";
            
            // Try matching full path first
            if (Regex.IsMatch(filePath, regexPattern, RegexOptions.IgnoreCase))
                return true;
            
            // Also try matching just the filename
            var fileName = Path.GetFileName(filePath);
            if (Regex.IsMatch(fileName, regexPattern, RegexOptions.IgnoreCase))
                return true;
            
            return false;
        }
        
        /// <summary>
        /// Finds the first file that matches any of the given patterns.
        /// Patterns are tried in order; first match wins.
        /// </summary>
        public static string FindFirstMatch(List<RepoFile> files, string[] patterns)
        {
            if (files == null || patterns == null)
                return null;
            
            foreach (var pattern in patterns)
            {
                foreach (var file in files)
                {
                    if (MatchesPattern(file.Path, pattern))
                        return file.Path;
                }
            }
            
            return null;
        }
        
        /// <summary>
        /// Checks if all file definitions can be resolved from the file list.
        /// </summary>
        public static bool CanResolveAllFiles(List<RepoFile> files, FileDefinition[] fileDefinitions)
        {
            if (files == null || fileDefinitions == null)
                return false;
            
            foreach (var fileDef in fileDefinitions)
            {
                if (fileDef.Optional)
                    continue;
                
                var match = FindFirstMatch(files, fileDef.Patterns);
                if (match == null)
                    return false;
            }
            
            return true;
        }
        
        /// <summary>
        /// Resolves all file definitions to actual file paths.
        /// Returns a dictionary mapping file keys to matched paths.
        /// </summary>
        public static Dictionary<string, string> ResolveFiles(List<RepoFile> files, FileDefinition[] fileDefinitions)
        {
            var result = new Dictionary<string, string>();
            
            if (files == null || fileDefinitions == null)
                return result;
            
            foreach (var fileDef in fileDefinitions)
            {
                var match = FindFirstMatch(files, fileDef.Patterns);
                if (match != null)
                {
                    result[fileDef.Key] = match;
                    
                    // Auto-include companion .onnx_data file for ONNX models that use external data.
                    if (ModelFileUtility.IsOnnxModelPath(match))
                    {
                        var dataKey = fileDef.Key + "_data";
                        if (!result.ContainsKey(dataKey))
                        {
                            var dataPath = match + "_data";
                            if (files.Any(f => f.Path == dataPath))
                                result[dataKey] = dataPath;
                        }
                    }
                }
            }
            
            return result;
        }
        
        #region JSON Parsing
        
        private static List<RepoFile> ParseHuggingFaceTree(string json)
        {
            var files = new List<RepoFile>();
            
            try
            {
                // Model info API format: { "siblings": [{"rfilename": "onnx/encoder.onnx", "size": 1234}, ...] }
                var siblingsMatch = Regex.Match(json, @"""siblings""\s*:\s*\[");
                if (siblingsMatch.Success)
                {
                    var matches = Regex.Matches(json, @"""rfilename""\s*:\s*""([^""]+)""");
                    foreach (Match match in matches)
                    {
                        var path = match.Groups[1].Value;
                        // Try to find size near this entry
                        var sizeMatch = Regex.Match(json.Substring(match.Index, Math.Min(200, json.Length - match.Index)), @"""size""\s*:\s*(\d+)");
                        long size = sizeMatch.Success ? long.Parse(sizeMatch.Groups[1].Value) : 0;
                        files.Add(new RepoFile(path, Path.GetFileName(path), size));
                    }
                }
                
                // Fallback: tree API format [{"path": "file.onnx", "type": "file", "size": 1234}, ...]
                if (files.Count == 0)
                {
                    var matches = Regex.Matches(json, @"\{[^}]*""path""\s*:\s*""([^""]+)""[^}]*""type""\s*:\s*""file""[^}]*\}");
                    foreach (Match match in matches)
                    {
                        var path = match.Groups[1].Value;
                        var sizeMatch = Regex.Match(match.Value, @"""size""\s*:\s*(\d+)");
                        long size = sizeMatch.Success ? long.Parse(sizeMatch.Groups[1].Value) : 0;
                        files.Add(new RepoFile(path, Path.GetFileName(path), size));
                    }
                    
                    // Also try alternative format where type comes before path
                    if (files.Count == 0)
                    {
                        matches = Regex.Matches(json, @"\{[^}]*""type""\s*:\s*""file""[^}]*""path""\s*:\s*""([^""]+)""[^}]*\}");
                        foreach (Match match in matches)
                        {
                            var path = match.Groups[1].Value;
                            files.Add(new RepoFile(path, Path.GetFileName(path), 0));
                        }
                    }
                }
            }
            catch { }
            
            return files;
        }
        
        private static List<RepoFile> ParseGitHubTree(string json)
        {
            var files = new List<RepoFile>();
            
            try
            {
                // Format: [{"path": "file.txt", "type": "file", "size": 1234}, ...]
                var matches = Regex.Matches(json, @"\{[^}]*""path""\s*:\s*""([^""]+)""[^}]*""type""\s*:\s*""file""[^}]*\}");
                foreach (Match match in matches)
                {
                    var path = match.Groups[1].Value;
                    var sizeMatch = Regex.Match(match.Value, @"""size""\s*:\s*(\d+)");
                    long size = sizeMatch.Success ? long.Parse(sizeMatch.Groups[1].Value) : 0;
                    files.Add(new RepoFile(path, Path.GetFileName(path), size));
                }
            }
            catch { }
            
            return files;
        }
        
        private static List<RepoFile> ParseGitLabTree(string json)
        {
            var files = new List<RepoFile>();
            
            try
            {
                // Format: [{"path": "file.txt", "type": "blob", "name": "file.txt"}, ...]
                var matches = Regex.Matches(json, @"\{[^}]*""path""\s*:\s*""([^""]+)""[^}]*""type""\s*:\s*""blob""[^}]*\}");
                foreach (Match match in matches)
                {
                    var path = match.Groups[1].Value;
                    files.Add(new RepoFile(path, Path.GetFileName(path), 0));
                }
            }
            catch { }
            
            return files;
        }
        
        #endregion
    }
}
