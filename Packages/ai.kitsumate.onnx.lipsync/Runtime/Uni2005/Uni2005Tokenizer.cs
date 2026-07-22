using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace KitsuMate.Onnx.LipSync.Uni2005
{
    /// <summary>
    /// Tokenizer for Uni2005 Allosaurus phoneme vocabulary.
    /// </summary>
    public class Uni2005Tokenizer
    {
        private Dictionary<int, string> _idToPhone;
        private Dictionary<string, int> _phoneToId;
        private int _blankTokenId = 0;
        private bool _isInitialized;
        
        public bool IsInitialized => _isInitialized;
        public int PhoneCount => _idToPhone?.Count ?? 0;
        public int BlankTokenId => _blankTokenId;
        
        /// <summary>
        /// Initialize tokenizer from vocabulary JSON.
        /// Supports both formats: {"0": "phone", ...} (id→phone) and {"phone": 0, ...} (phone→id).
        /// </summary>
        public bool LoadVocabulary(string vocabJson)
        {
            _idToPhone = new Dictionary<int, string>();
            _phoneToId = new Dictionary<string, int>();
            
            try
            {
                var root = JObject.Parse(vocabJson);
                foreach (var property in root.Properties())
                {
                    int id;
                    string phone;
                    
                    if (int.TryParse(property.Name, out id))
                    {
                        // Format: {"0": "<blank>", "1": "a", ...}
                        phone = property.Value.Value<string>() ?? string.Empty;
                    }
                    else if (property.Value.Type == JTokenType.Integer && property.Value.Value<int?>() is int parsedId)
                    {
                        // Format: {"<blank>": 0, "a": 1, ...}
                        id = parsedId;
                        phone = property.Name;
                    }
                    else
                    {
                        continue;
                    }

                    if (string.IsNullOrWhiteSpace(phone))
                        continue;
                    
                    _idToPhone[id] = phone;
                    _phoneToId[phone] = id;
                    
                    // Find blank token
                    if (phone == "<blank>" || phone == "<blk>")
                    {
                        _blankTokenId = id;
                    }
                }
                
                _isInitialized = _idToPhone.Count > 0;
                
                if (_isInitialized)
                    Debug.Log($"[Uni2005Tokenizer] Loaded {_idToPhone.Count} phones, blank id: {_blankTokenId}");
                
                return _isInitialized;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Uni2005Tokenizer] Error loading vocabulary: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Get phoneme string for a token ID.
        /// </summary>
        public string GetPhone(int id)
        {
            return _idToPhone.TryGetValue(id, out var phone) ? phone : "<unk>";
        }

        /// <summary>
        /// Get token ID for a phoneme string.
        /// </summary>
        public int GetId(string phone)
        {
            return _phoneToId.TryGetValue(phone, out int id) ? id : _blankTokenId;
        }

        /// <summary>
        /// Check if a token ID represents the blank token.
        /// </summary>
        public bool IsBlank(int id) => id == _blankTokenId;

        /// <summary>
        /// Get all phoneme strings.
        /// </summary>
        public IEnumerable<string> GetAllPhones() => _idToPhone.Values;
    }
}
