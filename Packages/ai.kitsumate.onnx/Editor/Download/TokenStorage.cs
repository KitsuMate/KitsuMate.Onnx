using System;
using System.Text;
using UnityEditor;

namespace KitsuMate.Onnx.Editor.Download
{
    /// <summary>
    /// Stores and retrieves personal access tokens per domain using XOR encryption.
    /// Tokens are persisted in EditorPrefs.
    /// </summary>
    public static class TokenStorage
    {
        private const string EncryptionKey = "KitsuMate";
        private const string PrefsKeyPrefix = "KitsuMate.Token.";
        
        /// <summary>
        /// Saves an encrypted token for the specified domain.
        /// </summary>
        public static void SaveToken(string domain, string token)
        {
            if (string.IsNullOrEmpty(domain))
                return;
                
            var encrypted = Encrypt(token);
            EditorPrefs.SetString(PrefsKeyPrefix + domain, encrypted);
        }
        
        /// <summary>
        /// Loads and decrypts a token for the specified domain.
        /// Returns null if no token is stored.
        /// </summary>
        public static string LoadToken(string domain)
        {
            if (string.IsNullOrEmpty(domain))
                return null;
                
            var encrypted = EditorPrefs.GetString(PrefsKeyPrefix + domain, null);
            if (string.IsNullOrEmpty(encrypted))
                return null;
                
            return Decrypt(encrypted);
        }
        
        /// <summary>
        /// Clears the stored token for the specified domain.
        /// </summary>
        public static void ClearToken(string domain)
        {
            if (string.IsNullOrEmpty(domain))
                return;
                
            EditorPrefs.DeleteKey(PrefsKeyPrefix + domain);
        }
        
        /// <summary>
        /// Extracts the domain from a URL.
        /// </summary>
        public static string ExtractDomain(string url)
        {
            if (string.IsNullOrEmpty(url))
                return null;
                
            try
            {
                // Handle owner/repo format (assume HuggingFace)
                if (!url.Contains("://") && url.Contains("/") && !url.Contains("."))
                {
                    return "huggingface.co";
                }
                
                var uri = new Uri(url);
                return uri.Host;
            }
            catch
            {
                return null;
            }
        }
        
        /// <summary>
        /// Encrypts a string using XOR cipher with the encryption key, then Base64 encodes.
        /// </summary>
        private static string Encrypt(string plainText)
        {
            if (string.IsNullOrEmpty(plainText))
                return string.Empty;
                
            var plainBytes = Encoding.UTF8.GetBytes(plainText);
            var keyBytes = Encoding.UTF8.GetBytes(EncryptionKey);
            var result = new byte[plainBytes.Length];
            
            for (int i = 0; i < plainBytes.Length; i++)
            {
                result[i] = (byte)(plainBytes[i] ^ keyBytes[i % keyBytes.Length]);
            }
            
            return Convert.ToBase64String(result);
        }
        
        /// <summary>
        /// Decrypts a Base64 encoded XOR encrypted string.
        /// </summary>
        private static string Decrypt(string encrypted)
        {
            if (string.IsNullOrEmpty(encrypted))
                return string.Empty;
                
            try
            {
                var encryptedBytes = Convert.FromBase64String(encrypted);
                var keyBytes = Encoding.UTF8.GetBytes(EncryptionKey);
                var result = new byte[encryptedBytes.Length];
                
                for (int i = 0; i < encryptedBytes.Length; i++)
                {
                    result[i] = (byte)(encryptedBytes[i] ^ keyBytes[i % keyBytes.Length]);
                }
                
                return Encoding.UTF8.GetString(result);
            }
            catch
            {
                return string.Empty;
            }
        }
    }
}
