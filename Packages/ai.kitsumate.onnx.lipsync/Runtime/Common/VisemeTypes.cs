using System;
using System.Collections.Generic;
using UnityEngine;

namespace KitsuMate.Onnx.LipSync
{
    /// <summary>
    /// Standard viseme types for lip sync animation.
    /// Based on the Oculus/Meta viseme set.
    /// </summary>
    public enum Viseme
    {
        /// <summary>Silent/neutral position.</summary>
        sil = 0,
        
        /// <summary>Bilabial plosives: p, b, m</summary>
        PP = 1,
        
        /// <summary>Labiodental fricatives: f, v</summary>
        FF = 2,
        
        /// <summary>Dental fricatives: th (θ, ð)</summary>
        TH = 3,
        
        /// <summary>Alveolar plosives: t, d</summary>
        DD = 4,
        
        /// <summary>Velar plosives: k, g</summary>
        kk = 5,
        
        /// <summary>Postalveolar: ch, j, sh, zh</summary>
        CH = 6,
        
        /// <summary>Alveolar fricatives: s, z</summary>
        SS = 7,
        
        /// <summary>Alveolar nasal: n</summary>
        nn = 8,
        
        /// <summary>Rhotic: r</summary>
        RR = 9,
        
        /// <summary>Open vowel: a, ah</summary>
        aa = 10,
        
        /// <summary>Mid-front vowel: e, eh</summary>
        E = 11,
        
        /// <summary>High-front vowel: i, ih</summary>
        ih = 12,
        
        /// <summary>Mid-back vowel: o, oh</summary>
        oh = 13,
        
        /// <summary>High-back vowel: u, oo</summary>
        ou = 14
    }
    
    /// <summary>
    /// A single viseme with timing information.
    /// </summary>
    [Serializable]
    public struct VisemeFrame
    {
        public Viseme Viseme;
        public float StartTime;
        public float EndTime;
        public float Weight;
        
        public float Duration => EndTime - StartTime;
        
        public VisemeFrame(Viseme viseme, float startTime, float endTime, float weight = 1f)
        {
            Viseme = viseme;
            StartTime = startTime;
            EndTime = endTime;
            Weight = weight;
        }
        
        public override string ToString() => 
            $"[{StartTime:F3}s - {EndTime:F3}s] {Viseme} ({Weight:F2})";
    }
    
    /// <summary>
    /// A phoneme with timing information.
    /// </summary>
    [Serializable]
    public struct PhonemeFrame
    {
        public string Phoneme;
        public float StartTime;
        public float EndTime;
        public float Probability;
        public float Rms;
        
        public float Duration => EndTime - StartTime;
        
        public PhonemeFrame(string phoneme, float startTime, float endTime, float probability = 1f, float rms = 0f)
        {
            Phoneme = phoneme;
            StartTime = startTime;
            EndTime = endTime;
            Probability = probability;
            Rms = rms;
        }
        
        public override string ToString() => 
            $"[{StartTime:F3}s - {EndTime:F3}s] '{Phoneme}' (p={Probability:F3}, rms={Rms:F4})";
    }
    
    /// <summary>
    /// Timeline of visemes for lip sync animation.
    /// </summary>
    [Serializable]
    public class VisemeTimeline
    {
        [SerializeField] private List<VisemeFrame> _frames = new();
        [SerializeField] private float _duration;
        
        /// <summary>All viseme frames in the timeline.</summary>
        public List<VisemeFrame> Frames => _frames;
        
        /// <summary>Total duration of the timeline in seconds.</summary>
        public float Duration
        {
            get => _duration;
            set => _duration = value;
        }
        
        /// <summary>Number of frames in the timeline.</summary>
        public int FrameCount => _frames.Count;
        
        /// <summary>
        /// Get the viseme at a specific time.
        /// </summary>
        public VisemeFrame? GetFrameAtTime(float time)
        {
            foreach (var frame in _frames)
            {
                if (time >= frame.StartTime && time < frame.EndTime)
                    return frame;
            }
            return null;
        }
        
        /// <summary>
        /// Get interpolated viseme weights at a specific time.
        /// Returns a dictionary of viseme to weight (0-1).
        /// </summary>
        public Dictionary<Viseme, float> GetWeightsAtTime(float time, float blendDuration = 0.05f)
        {
            var weights = new Dictionary<Viseme, float>();
            
            foreach (Viseme v in Enum.GetValues(typeof(Viseme)))
            {
                weights[v] = 0f;
            }
            
            foreach (var frame in _frames)
            {
                float weight = 0f;
                
                if (time >= frame.StartTime && time < frame.EndTime)
                {
                    float blendIn = Mathf.Clamp01((time - frame.StartTime) / blendDuration);
                    float blendOut = Mathf.Clamp01((frame.EndTime - time) / blendDuration);
                    weight = Mathf.Min(blendIn, blendOut) * frame.Weight;
                }
                
                weights[frame.Viseme] = Mathf.Max(weights[frame.Viseme], weight);
            }
            
            return weights;
        }
        
        /// <summary>
        /// Add a frame to the timeline.
        /// </summary>
        public void AddFrame(VisemeFrame frame)
        {
            _frames.Add(frame);
            if (frame.EndTime > _duration)
                _duration = frame.EndTime;
        }
        
        /// <summary>
        /// Clear all frames.
        /// </summary>
        public void Clear()
        {
            _frames.Clear();
            _duration = 0f;
        }
        
        /// <summary>
        /// Sort frames by start time.
        /// </summary>
        public void Sort()
        {
            _frames.Sort((a, b) => a.StartTime.CompareTo(b.StartTime));
        }
    }
}
