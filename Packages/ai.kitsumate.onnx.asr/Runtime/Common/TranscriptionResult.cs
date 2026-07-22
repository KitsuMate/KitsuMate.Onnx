using System;
using System.Collections.Generic;
using UnityEngine;

namespace KitsuMate.Onnx.Asr
{
    /// <summary>
    /// Represents a word with timing information from transcription.
    /// </summary>
    [Serializable]
    public class WordTimestamp
    {
        [SerializeField] private string _word;
        [SerializeField] private float _startTime;
        [SerializeField] private float _endTime;
        [SerializeField] private float _probability;

        public string Word => _word;
        public float StartTime => _startTime;
        public float EndTime => _endTime;
        public float Probability => _probability;
        public float Duration => _endTime - _startTime;

        public WordTimestamp(string word, float startTime, float endTime, float probability = 1.0f)
        {
            _word = word;
            _startTime = startTime;
            _endTime = endTime;
            _probability = probability;
        }

        public override string ToString() => $"[{_startTime:F2}s - {_endTime:F2}s] {_word} (p={_probability:F3})";
    }

    /// <summary>
    /// Represents a single character with timing data.
    /// </summary>
    [Serializable]
    public class CharacterTimestamp
    {
        [SerializeField] private string _character;
        [SerializeField] private float _startTime;
        [SerializeField] private float _endTime;
        [SerializeField] private float _probability;

        public string Character => _character;
        public float StartTime => _startTime;
        public float EndTime => _endTime;
        public float Probability => _probability;

        public CharacterTimestamp(string character, float startTime, float endTime, float probability = 1.0f)
        {
            _character = character;
            _startTime = startTime;
            _endTime = endTime;
            _probability = probability;
        }

        public override string ToString() => $"[{_startTime:F2}s - {_endTime:F2}s] '{_character}' (p={_probability:F3})";
    }

    /// <summary>
    /// Result of transcription with optional timing information.
    /// </summary>
    [Serializable]
    public class TranscriptionResult
    {
        [SerializeField] private string _text = "";
        [SerializeField] private string _language = "";
        [SerializeField] private float _duration;
        [SerializeField] private List<WordTimestamp> _wordTimestamps = new();
        [SerializeField] private List<CharacterTimestamp> _characterTimestamps = new();
        [SerializeField] private List<int> _tokenIds = new();

        /// <summary>Full transcribed text.</summary>
        public string Text
        {
            get => _text;
            set => _text = value;
        }

        /// <summary>Detected or specified language code (e.g., "en", "ja").</summary>
        public string Language
        {
            get => _language;
            set => _language = value;
        }

        /// <summary>Duration of the audio in seconds.</summary>
        public float Duration
        {
            get => _duration;
            set => _duration = value;
        }

        /// <summary>Word-level timing information.</summary>
        public List<WordTimestamp> WordTimestamps
        {
            get => _wordTimestamps;
            set => _wordTimestamps = value ?? new List<WordTimestamp>();
        }

        /// <summary>Character-level timing information.</summary>
        public List<CharacterTimestamp> CharacterTimestamps => _characterTimestamps;

        /// <summary>Token IDs from the model.</summary>
        public List<int> TokenIds => _tokenIds;

        /// <summary>Whether word timestamps are available.</summary>
        public bool HasWordTimestamps => _wordTimestamps.Count > 0;

        /// <summary>Whether character timestamps are available.</summary>
        public bool HasCharacterTimestamps => _characterTimestamps.Count > 0;

        public TranscriptionResult() { }

        public TranscriptionResult(string text, string language = null)
        {
            _text = text ?? "";
            _language = language ?? "";
        }

        /// <summary>
        /// Get the word at a specific time.
        /// </summary>
        public WordTimestamp GetWordAtTime(float time)
        {
            foreach (var word in _wordTimestamps)
            {
                if (time >= word.StartTime && time <= word.EndTime)
                    return word;
            }
            return null;
        }

        /// <summary>
        /// Get all words in a time range.
        /// </summary>
        public IEnumerable<WordTimestamp> GetWordsInRange(float startTime, float endTime)
        {
            foreach (var word in _wordTimestamps)
            {
                if (word.EndTime >= startTime && word.StartTime <= endTime)
                    yield return word;
            }
        }

        public override string ToString() => _text;
    }

    /// <summary>
    /// Segment of audio with voice activity.
    /// </summary>
    [Serializable]
    public struct VoiceSegment
    {
        public float StartTime;
        public float EndTime;
        public float Duration => EndTime - StartTime;

        public VoiceSegment(float startTime, float endTime)
        {
            StartTime = startTime;
            EndTime = endTime;
        }
    }
}
