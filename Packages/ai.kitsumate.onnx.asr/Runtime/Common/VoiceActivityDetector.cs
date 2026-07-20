using System;
using System.Collections.Generic;
using UnityEngine;

namespace KitsuMate.Onnx.Asr
{
    /// <summary>
    /// Detects voice activity in audio samples.
    /// Used for segmenting audio before transcription.
    /// </summary>
    public static class VoiceActivityDetector
    {
        /// <summary>
        /// Detect voice segments in audio samples.
        /// </summary>
        /// <param name="samples">Audio samples (mono, normalized -1 to 1)</param>
        /// <param name="sampleRate">Sample rate in Hz</param>
        /// <param name="volumeThreshold">Minimum volume to consider as voice (0-1)</param>
        /// <param name="minSegmentDuration">Minimum segment duration in seconds</param>
        /// <param name="maxSegmentGap">Maximum gap between segments to merge</param>
        /// <returns>List of voice segments with start/end times</returns>
        public static List<VoiceSegment> DetectVoiceSegments(
            float[] samples,
            int sampleRate,
            float volumeThreshold = 0.02f,
            float minSegmentDuration = 0.1f,
            float maxSegmentGap = 0.3f)
        {
            var segments = new List<VoiceSegment>();
            
            // Window size for RMS calculation (20ms)
            int windowSize = sampleRate / 50;
            int hopSize = windowSize / 2;
            
            float? segmentStart = null;
            float lastVoiceTime = -1f;
            
            for (int i = 0; i < samples.Length - windowSize; i += hopSize)
            {
                float rms = CalculateRMS(samples, i, windowSize);
                float currentTime = (float)i / sampleRate;
                
                bool isVoice = rms > volumeThreshold;
                
                if (isVoice)
                {
                    if (segmentStart == null)
                    {
                        // Check if we should merge with previous segment
                        if (segments.Count > 0 && currentTime - lastVoiceTime <= maxSegmentGap)
                        {
                            // Remove last segment, we'll extend it
                            var lastSegment = segments[^1];
                            segments.RemoveAt(segments.Count - 1);
                            segmentStart = lastSegment.StartTime;
                        }
                        else
                        {
                            segmentStart = currentTime;
                        }
                    }
                    lastVoiceTime = currentTime;
                }
                else if (segmentStart != null)
                {
                    // Check if gap is too long
                    if (currentTime - lastVoiceTime > maxSegmentGap)
                    {
                        float duration = lastVoiceTime - segmentStart.Value;
                        if (duration >= minSegmentDuration)
                        {
                            segments.Add(new VoiceSegment(segmentStart.Value, lastVoiceTime));
                        }
                        segmentStart = null;
                    }
                }
            }
            
            // Handle final segment
            if (segmentStart != null)
            {
                float endTime = Math.Max(lastVoiceTime, (float)samples.Length / sampleRate);
                float duration = endTime - segmentStart.Value;
                if (duration >= minSegmentDuration)
                {
                    segments.Add(new VoiceSegment(segmentStart.Value, endTime));
                }
            }
            
            return segments;
        }
        
        /// <summary>
        /// Detect voice segments from an AudioClip.
        /// </summary>
        public static List<VoiceSegment> DetectVoiceSegments(
            AudioClip audioClip,
            float volumeThreshold = 0.02f,
            float minSegmentDuration = 0.1f,
            float maxSegmentGap = 0.3f)
        {
            var samples = new float[audioClip.samples * audioClip.channels];
            audioClip.GetData(samples, 0);
            
            // Convert to mono if stereo
            if (audioClip.channels > 1)
            {
                samples = ConvertToMono(samples, audioClip.channels);
            }
            
            return DetectVoiceSegments(samples, audioClip.frequency, volumeThreshold, minSegmentDuration, maxSegmentGap);
        }
        
        private static float CalculateRMS(float[] samples, int start, int length)
        {
            float sum = 0f;
            int end = Math.Min(start + length, samples.Length);
            int count = end - start;
            
            for (int i = start; i < end; i++)
            {
                sum += samples[i] * samples[i];
            }
            
            return Mathf.Sqrt(sum / count);
        }
        
        private static float[] ConvertToMono(float[] samples, int channels)
        {
            int monoLength = samples.Length / channels;
            var mono = new float[monoLength];
            
            for (int i = 0; i < monoLength; i++)
            {
                float sum = 0f;
                for (int c = 0; c < channels; c++)
                {
                    sum += samples[i * channels + c];
                }
                mono[i] = sum / channels;
            }
            
            return mono;
        }
    }
}
