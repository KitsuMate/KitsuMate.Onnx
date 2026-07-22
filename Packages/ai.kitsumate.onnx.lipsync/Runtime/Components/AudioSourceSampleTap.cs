using System;
using UnityEngine;

namespace KitsuMate.Onnx.LipSync
{
    /// <summary>
    /// Captures the most recent samples emitted by an AudioSource on the same GameObject.
    /// Intended for realtime lip sync analysis.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(AudioSource))]
    public sealed class AudioSourceSampleTap : MonoBehaviour
    {
        private readonly object _bufferLock = new();
        private float[] _buffer;
        private int _writeIndex;
        private int _sampleCount;

        /// <summary>Current sample rate used by the audio callback.</summary>
        public int SampleRate => AudioSettings.outputSampleRate;

        /// <summary>
        /// Ensure the internal circular buffer can hold the requested number of mono samples.
        /// </summary>
        public void Configure(int capacitySamples)
        {
            capacitySamples = Mathf.Max(1, capacitySamples);

            lock (_bufferLock)
            {
                if (_buffer != null && _buffer.Length == capacitySamples)
                    return;

                _buffer = new float[capacitySamples];
                _writeIndex = 0;
                _sampleCount = 0;
            }
        }

        /// <summary>
        /// Clear all captured samples.
        /// </summary>
        public void Clear()
        {
            lock (_bufferLock)
            {
                if (_buffer == null)
                    return;

                Array.Clear(_buffer, 0, _buffer.Length);
                _writeIndex = 0;
                _sampleCount = 0;
            }
        }

        /// <summary>
        /// Copy the newest mono window into the provided buffer.
        /// </summary>
        public bool TryCopyLatestWindow(float[] destination, int sampleCount, out int sampleRate, out float rms)
        {
            sampleRate = SampleRate;
            rms = 0f;

            if (destination == null)
                throw new ArgumentNullException(nameof(destination));

            lock (_bufferLock)
            {
                if (_buffer == null || sampleCount <= 0 || destination.Length < sampleCount || _sampleCount < sampleCount)
                    return false;

                int start = _writeIndex - sampleCount;
                if (start < 0)
                    start += _buffer.Length;

                double sumSquares = 0d;
                int copied = 0;
                for (int i = 0; i < sampleCount; i++)
                {
                    int index = start + i;
                    if (index >= _buffer.Length)
                        index -= _buffer.Length;

                    float sample = _buffer[index];
                    destination[i] = sample;
                    sumSquares += sample * sample;
                    copied++;
                }

                rms = copied > 0 ? Mathf.Sqrt((float)(sumSquares / copied)) : 0f;
                return true;
            }
        }

        private void OnDisable()
        {
            Clear();
        }

        private void OnAudioFilterRead(float[] data, int channels)
        {
            if (_buffer == null || data == null || data.Length == 0)
                return;

            channels = Mathf.Max(1, channels);
            int frameCount = data.Length / channels;

            lock (_bufferLock)
            {
                if (_buffer == null)
                    return;

                for (int frame = 0; frame < frameCount; frame++)
                {
                    float sum = 0f;
                    int offset = frame * channels;
                    for (int channel = 0; channel < channels; channel++)
                    {
                        sum += data[offset + channel];
                    }

                    WriteSample(sum / channels);
                }
            }
        }

        private void WriteSample(float sample)
        {
            _buffer[_writeIndex] = sample;
            _writeIndex = (_writeIndex + 1) % _buffer.Length;
            if (_sampleCount < _buffer.Length)
                _sampleCount++;
        }
    }
}
