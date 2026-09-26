using System;
using System.Collections.Generic;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    /// <summary>
    /// The four speech-encoder outputs that condition one reference voice. They depend only on the
    /// reference audio, so they can be reused for every request that speaks with that voice.
    /// </summary>
    public sealed class ChatterboxVoiceConditioning : IDisposable
    {
        /// <summary>Conditioning prefix prepended to the language model input (audio_features).</summary>
        public OnnxTensor Features { get; }

        /// <summary>Speech tokens of the reference, prepended before decoding (audio_tokens).</summary>
        public OnnxTensor Tokens { get; }

        /// <summary>Speaker vector for the decoder (speaker_embeddings).</summary>
        public OnnxTensor Speaker { get; }

        /// <summary>Reference mel features for the decoder (speaker_features).</summary>
        public OnnxTensor SpeakerFeatures { get; }

        public ChatterboxVoiceConditioning(OnnxTensor features, OnnxTensor tokens, OnnxTensor speaker,
            OnnxTensor speakerFeatures)
        {
            Features = features;
            Tokens = tokens;
            Speaker = speaker;
            SpeakerFeatures = speakerFeatures;
        }

        public void Dispose()
        {
            Features?.Dispose();
            Tokens?.Dispose();
            Speaker?.Dispose();
            SpeakerFeatures?.Dispose();
        }
    }

    /// <summary>
    /// Least-recently-used cache of encoded reference voices, keyed by AudioClip. Encoding a voice
    /// costs about as much as generating a second of speech, and a scene usually reuses a handful of
    /// voices, so caching removes that cost from every request after the first.
    /// </summary>
    public sealed class ChatterboxVoiceCache : IDisposable
    {
        private readonly int _capacity;
        private readonly LinkedList<(EntityId Key, ChatterboxVoiceConditioning Voice)> _order = new();
        private readonly Dictionary<EntityId, LinkedListNode<(EntityId Key, ChatterboxVoiceConditioning Voice)>> _byKey = new();

        public ChatterboxVoiceCache(int capacity) => _capacity = Math.Max(1, capacity);

        public bool Contains(EntityId clip) => _byKey.ContainsKey(clip);

        public bool TryGet(EntityId clip, out ChatterboxVoiceConditioning voice)
        {
            if (_byKey.TryGetValue(clip, out var node))
            {
                _order.Remove(node);
                _order.AddFirst(node);
                voice = node.Value.Voice;
                return true;
            }
            voice = null;
            return false;
        }

        public void Put(EntityId clip, ChatterboxVoiceConditioning voice)
        {
            if (_byKey.TryGetValue(clip, out var existing))
            {
                existing.Value.Voice.Dispose();
                _order.Remove(existing);
                _byKey.Remove(clip);
            }
            else if (_byKey.Count >= _capacity)
            {
                var last = _order.Last;
                last.Value.Voice.Dispose();
                _byKey.Remove(last.Value.Key);
                _order.RemoveLast();
            }

            _byKey[clip] = _order.AddFirst((clip, voice));
        }

        public void Dispose()
        {
            foreach (var item in _order) item.Voice.Dispose();
            _order.Clear();
            _byKey.Clear();
        }
    }
}
