using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
using KitsuMate.Onnx;

namespace KitsuMate.Onnx.LipSync.Uni2005
{
    /// <summary>
    /// Uni2005 Allosaurus-based phoneme detection engine.
    /// Detects phonemes from audio using MFCC features and maps them to visemes.
    /// </summary>
    public sealed class Uni2005EngineRuntime : LipSyncEngineRuntime
    {
        [Header("Model Set")]
        private readonly Uni2005ModelSet _modelSet;
        
        [Header("Detection Settings")]
        [SerializeField, Range(64, 2048), Tooltip("Frames per inference batch")]
        private int _framesPerBatch = 512;
        
        [SerializeField, Range(0f, 1f), Tooltip("Volume threshold for silence detection")]
        private float _volumeThreshold = 0.01f;
        
        // Runtime state
        private IOnnxSession _session;
        private Uni2005Tokenizer _tokenizer;
        private IPhonemeToVisemeMap _phonemeMap;
        private float[] _cachedSamples;
        
        /// <summary>Model set containing Uni2005 models.</summary>
        public Uni2005EngineRuntime(Uni2005ModelSet modelSet, int framesPerBatch, float volumeThreshold, bool verbose)
        {
            _modelSet = modelSet;
            _framesPerBatch = framesPerBatch;
            _volumeThreshold = volumeThreshold;
            VerboseLogging = verbose;
        }
        
        /// <summary>Volume threshold for silence detection.</summary>
        public float VolumeThreshold
        {
            get => _volumeThreshold;
            set => _volumeThreshold = Mathf.Clamp01(value);
        }
        
        public override IPhonemeToVisemeMap PhonemeMap => 
            _phonemeMap ??= new StandardPhonemeToVisemeMap();

        protected override void OnLoadMainThread(OnnxBackend backend)
        {
            if (_modelSet == null || !_modelSet.IsComplete)
                throw new InvalidOperationException("Uni2005ModelSet is not assigned or incomplete");

            _tokenizer = new Uni2005Tokenizer();
            _tokenizer.LoadVocabulary(_modelSet.Vocabulary.text);
            _phonemeMap = new StandardPhonemeToVisemeMap();
        }
        
        protected override void OnLoadBackground(OnnxBackend backend)
        {
            if (_modelSet == null || !_modelSet.IsComplete)
                throw new InvalidOperationException("Uni2005ModelSet is not assigned or incomplete");
            
            _session = backend.CreateSession(_modelSet.AcousticModel);
            
            if (VerboseLogging)
                Debug.Log($"[Uni2005Engine] Loaded with {_tokenizer.PhoneCount} phones");
        }
        
        protected override void OnUnload()
        {
            _session?.Dispose();
            _session = null;
            _tokenizer = null;
            _phonemeMap = null;
        }
        
        protected override void OnPrepareInput(LipSyncRequest input)
        {
            _cachedSamples = input.Audio != null ? ExtractAndPrepareSamples(input.Audio) : null;
        }
        
        protected override VisemeTimeline DetectVisemesFromPreparedAudio(AudioClip input)
        {
            var samples = _cachedSamples;
            _cachedSamples = null;
            var phonemes = samples != null
                ? DetectPhonemesFromSamples(samples, Uni2005Constants.SampleRate)
                : DetectPhonemes(input);
            
            if (VerboseLogging)
                Debug.Log($"[Uni2005Engine] Detected {phonemes.Count} phonemes from {samples?.Length ?? 0} samples");
            
            return PhonemesToVisemes(phonemes);
        }
        
        protected override VisemeTimeline DetectVisemes(float[] samples, int sampleRate)
        {
            var phonemes = DetectPhonemesFromSamples(samples, sampleRate);
            return PhonemesToVisemes(phonemes);
        }
        
        protected override List<PhonemeFrame> DetectPhonemes(AudioClip audioClip)
        {
            var samples = ExtractAndPrepareSamples(audioClip);
            return DetectPhonemesFromSamples(samples, Uni2005Constants.SampleRate);
        }
        
        private List<PhonemeFrame> DetectPhonemesFromSamples(float[] samples, int sampleRate)
        {
            if (sampleRate != Uni2005Constants.SampleRate)
                samples = Resample(samples, sampleRate, Uni2005Constants.SampleRate);
            
            var mfccFeatures = ExtractMfccFeatures(samples);
            
            if (VerboseLogging)
                Debug.Log($"[Uni2005Engine] MFCC features: {mfccFeatures.GetLength(0)} frames x {mfccFeatures.GetLength(1)} coeffs from {samples.Length} samples ({(float)samples.Length / sampleRate:F2}s)");
            
            var phoneIds = RunInference(mfccFeatures, samples);
            
            if (VerboseLogging)
            {
                int nonBlank = phoneIds.Count(x => !_tokenizer.IsBlank(x.id));
                Debug.Log($"[Uni2005Engine] Inference: {phoneIds.Count} frames, {nonBlank} non-blank");
            }
            
            return DecodePhonemesWithTimings(phoneIds, samples.Length);
        }
        
        private float[] ExtractAndPrepareSamples(AudioClip clip)
        {
            var samples = new float[clip.samples * clip.channels];
            clip.GetData(samples, 0);
            
            // Convert to mono
            if (clip.channels > 1)
            {
                var mono = new float[clip.samples];
                for (int i = 0; i < clip.samples; i++)
                {
                    float sum = 0;
                    for (int c = 0; c < clip.channels; c++)
                        sum += samples[i * clip.channels + c];
                    mono[i] = sum / clip.channels;
                }
                samples = mono;
            }
            
            // Resample to 8kHz
            if (clip.frequency != Uni2005Constants.SampleRate)
            {
                samples = Resample(samples, clip.frequency, Uni2005Constants.SampleRate);
            }
            
            return samples;
        }
        
        private float[,] ExtractMfccFeatures(float[] samples)
        {
            int frameLength = (int)(Uni2005Constants.SampleRate * Uni2005Constants.FrameLengthSeconds);
            int frameStep = (int)(Uni2005Constants.SampleRate * Uni2005Constants.FrameStepSeconds);
            int numFrames = Math.Max(1, (samples.Length - frameLength) / frameStep + 1);
            int numCoeffs = Uni2005Constants.CepstralCoefficients;
            
            // Extract base MFCCs
            var baseMfcc = new float[numFrames, numCoeffs];
            
            // Pre-emphasis
            var emphasized = new float[samples.Length];
            emphasized[0] = samples[0];
            for (int i = 1; i < samples.Length; i++)
            {
                emphasized[i] = samples[i] - Uni2005Constants.PreEmphasis * samples[i - 1];
            }
            
            // Extract MFCC for each frame (parallelized — each frame is independent)
            Parallel.For(0, numFrames, frame =>
            {
                int start = frame * frameStep;
                var mfcc = ComputeMfccForFrame(emphasized, start, frameLength);
                
                for (int c = 0; c < numCoeffs && c < mfcc.Length; c++)
                {
                    baseMfcc[frame, c] = mfcc[c];
                }
            });
            
            // Speaker-level CMVN (cepstral mean and variance normalization)
            for (int c = 0; c < numCoeffs; c++)
            {
                float mean = 0;
                for (int f = 0; f < numFrames; f++)
                    mean += baseMfcc[f, c];
                mean /= numFrames;
                
                float variance = 0;
                for (int f = 0; f < numFrames; f++)
                {
                    float diff = baseMfcc[f, c] - mean;
                    variance += diff * diff;
                }
                variance /= numFrames;
                float stddev = MathF.Sqrt(variance + 1e-10f);
                
                for (int f = 0; f < numFrames; f++)
                {
                    baseMfcc[f, c] = (baseMfcc[f, c] - mean) / stddev;
                }
            }
            
            // 3-frame context windows [prev, curr, next] subsampled by 3
            int subsampledFrames = (numFrames + 2) / 3; // ceil(numFrames / 3)
            var features = new float[subsampledFrames, numCoeffs * 3];
            
            for (int i = 0; i < subsampledFrames; i++)
            {
                int center = i * 3;
                int prev = Math.Max(0, center - 1);
                int next = Math.Min(numFrames - 1, center + 1);
                
                for (int c = 0; c < numCoeffs; c++)
                {
                    features[i, c]                = baseMfcc[prev, c];
                    features[i, numCoeffs + c]     = baseMfcc[center, c];
                    features[i, numCoeffs * 2 + c] = baseMfcc[next, c];
                }
            }
            
            return features;
        }
        
        private float[] ComputeMfccForFrame(float[] samples, int start, int length)
        {
            // Extract frame with Hamming window
            var frame = new float[length];
            for (int i = 0; i < length && start + i < samples.Length; i++)
            {
                float window = 0.54f - 0.46f * MathF.Cos(2f * MathF.PI * i / (length - 1));
                frame[i] = samples[start + i] * window;
            }
            
            // Compute power spectrum (simplified FFT -> power)
            var powerSpectrum = ComputePowerSpectrum(frame);
            
            // Apply mel filterbank
            var melEnergies = ApplyMelFilterbank(powerSpectrum);
            
            // Log and DCT to get MFCC
            var mfcc = new float[Uni2005Constants.CepstralCoefficients];
            for (int i = 0; i < melEnergies.Length; i++)
            {
                melEnergies[i] = MathF.Log(MathF.Max(melEnergies[i], 1e-10f));
            }
            
            // DCT-II
            for (int c = 0; c < Uni2005Constants.CepstralCoefficients; c++)
            {
                float sum = 0;
                for (int m = 0; m < Uni2005Constants.MelBanks; m++)
                {
                    sum += melEnergies[m] * MathF.Cos(MathF.PI * c * (m + 0.5f) / Uni2005Constants.MelBanks);
                }
                mfcc[c] = sum;
            }
            
            // Cepstral liftering
            for (int c = 0; c < mfcc.Length; c++)
            {
                float lifter = 1f + (Uni2005Constants.CeplifterCoeff / 2f) * 
                    MathF.Sin(MathF.PI * c / Uni2005Constants.CeplifterCoeff);
                mfcc[c] *= lifter;
            }
            
            return mfcc;
        }
        
        private float[] ComputePowerSpectrum(float[] frame)
        {
            int nfft = 512;
            var spectrum = new float[nfft / 2 + 1];
            
            // Power spectrum using DFT
            for (int k = 0; k <= nfft / 2; k++)
            {
                float real = 0, imag = 0;
                for (int n = 0; n < frame.Length && n < nfft; n++)
                {
                    float angle = -2f * MathF.PI * k * n / nfft;
                    real += frame[n] * MathF.Cos(angle);
                    imag += frame[n] * MathF.Sin(angle);
                }
                spectrum[k] = (real * real + imag * imag) / nfft;
            }
            
            return spectrum;
        }
        
        private float[] ApplyMelFilterbank(float[] powerSpectrum)
        {
            var melEnergies = new float[Uni2005Constants.MelBanks];
            
            float lowMel = HzToMel(Uni2005Constants.LowFrequency);
            float highMel = HzToMel(Uni2005Constants.HighFrequency);
            
            int nfft = (powerSpectrum.Length - 1) * 2;
            
            for (int m = 0; m < Uni2005Constants.MelBanks; m++)
            {
                float melCenter = lowMel + (highMel - lowMel) * (m + 1) / (Uni2005Constants.MelBanks + 1);
                float melLeft = lowMel + (highMel - lowMel) * m / (Uni2005Constants.MelBanks + 1);
                float melRight = lowMel + (highMel - lowMel) * (m + 2) / (Uni2005Constants.MelBanks + 1);
                
                float centerHz = MelToHz(melCenter);
                float leftHz = MelToHz(melLeft);
                float rightHz = MelToHz(melRight);
                
                int centerBin = (int)(centerHz * nfft / Uni2005Constants.SampleRate);
                int leftBin = (int)(leftHz * nfft / Uni2005Constants.SampleRate);
                int rightBin = (int)(rightHz * nfft / Uni2005Constants.SampleRate);
                
                float energy = 0;
                for (int k = leftBin; k <= rightBin && k < powerSpectrum.Length; k++)
                {
                    float weight;
                    if (k <= centerBin)
                        weight = (k - leftBin) / (float)(centerBin - leftBin + 1);
                    else
                        weight = (rightBin - k) / (float)(rightBin - centerBin + 1);
                    
                    weight = MathF.Max(0, weight);
                    energy += powerSpectrum[k] * weight;
                }
                
                melEnergies[m] = energy;
            }
            
            return melEnergies;
        }
        
        private static float HzToMel(float hz) => 2595f * MathF.Log10(1f + hz / 700f);
        private static float MelToHz(float mel) => 700f * (MathF.Pow(10f, mel / 2595f) - 1f);
        
        private List<(int id, float rms)> RunInference(float[,] features, float[] samples)
        {
            var results = new List<(int, float)>();
            int numFrames = features.GetLength(0);
            int numCoeffs = features.GetLength(1);
            
            int baseFrameStep = (int)(Uni2005Constants.SampleRate * Uni2005Constants.FrameStepSeconds);
            int subsampledStep = baseFrameStep * 3;
            
            for (int batchStart = 0; batchStart < numFrames; batchStart += _framesPerBatch)
            {
                int batchSize = Math.Min(_framesPerBatch, numFrames - batchStart);
                
                var inputData = new float[1 * batchSize * numCoeffs];
                for (int f = 0; f < batchSize; f++)
                    for (int c = 0; c < numCoeffs; c++)
                        inputData[f * numCoeffs + c] = features[batchStart + f, c];
                
                var inputs = new Dictionary<string, OnnxTensor>
                {
                    ["mfcc"] = OnnxTensor.FromArray(inputData, new[] { 1, batchSize, numCoeffs }, "mfcc")
                };
                
                var outputs = _session.Run(inputs);
                var logits = outputs.Values.First().AsFloatArray();
                
                int vocabSize = _tokenizer.PhoneCount;
                for (int f = 0; f < batchSize; f++)
                {
                    int sampleStart = (batchStart + f) * subsampledStep;
                    float rms = ComputeRms(samples, sampleStart, subsampledStep);
                    
                    int bestId = 0;
                    float bestScore = float.MinValue;
                    for (int v = 0; v < vocabSize; v++)
                    {
                        float score = logits[f * vocabSize + v];
                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestId = v;
                        }
                    }
                    
                    results.Add((bestId, rms));
                }
            }
            
            return results;
        }
        
        private List<PhonemeFrame> DecodePhonemesWithTimings(List<(int id, float rms)> frameResults, int totalSamples)
        {
            var phonemes = new List<PhonemeFrame>();
            
            float frameDuration = Uni2005Constants.FrameStepSeconds * 3f;
            int? lastPhoneId = null;
            float segmentStart = 0;
            float rmsSum = 0f;
            int rmsCount = 0;
            
            for (int i = 0; i < frameResults.Count; i++)
            {
                var (phoneId, rms) = frameResults[i];
                float frameTime = i * frameDuration;
                
                if (rms < _volumeThreshold)
                    phoneId = _tokenizer.BlankTokenId;
                
                if (_tokenizer.IsBlank(phoneId))
                {
                    if (lastPhoneId.HasValue && !_tokenizer.IsBlank(lastPhoneId.Value))
                    {
                        float avgRms = rmsCount > 0 ? rmsSum / rmsCount : 0f;
                        phonemes.Add(new PhonemeFrame(
                            _tokenizer.GetPhone(lastPhoneId.Value),
                            segmentStart, frameTime, 1f, avgRms));
                    }
                    lastPhoneId = phoneId;
                    segmentStart = frameTime;
                    rmsSum = rms;
                    rmsCount = 1;
                }
                else if (!lastPhoneId.HasValue || phoneId != lastPhoneId.Value)
                {
                    if (lastPhoneId.HasValue && !_tokenizer.IsBlank(lastPhoneId.Value))
                    {
                        float avgRms = rmsCount > 0 ? rmsSum / rmsCount : 0f;
                        phonemes.Add(new PhonemeFrame(
                            _tokenizer.GetPhone(lastPhoneId.Value),
                            segmentStart, frameTime, 1f, avgRms));
                    }
                    lastPhoneId = phoneId;
                    segmentStart = frameTime;
                    rmsSum = rms;
                    rmsCount = 1;
                }
                else
                {
                    rmsSum += rms;
                    rmsCount++;
                }
            }
            
            if (lastPhoneId.HasValue && !_tokenizer.IsBlank(lastPhoneId.Value))
            {
                float endTime = frameResults.Count * frameDuration;
                float avgRms = rmsCount > 0 ? rmsSum / rmsCount : 0f;
                phonemes.Add(new PhonemeFrame(
                    _tokenizer.GetPhone(lastPhoneId.Value),
                    segmentStart, endTime, 1f, avgRms));
            }
            
            return phonemes;
        }
        
        private VisemeTimeline PhonemesToVisemes(List<PhonemeFrame> phonemes)
        {
            var timeline = new VisemeTimeline();
            Viseme lastNonSilViseme = Viseme.sil;
            
            foreach (var phoneme in phonemes)
            {
                var viseme = PhonemeMap.MapPhoneme(phoneme.Phoneme);
                
                if (viseme == Viseme.sil)
                {
                    if (phoneme.Rms >= _volumeThreshold && lastNonSilViseme != Viseme.sil)
                        viseme = lastNonSilViseme;
                    else if (phoneme.Duration < 0.1f)
                        continue;
                }
                
                if (viseme != Viseme.sil)
                    lastNonSilViseme = viseme;
                
                timeline.AddFrame(new VisemeFrame(
                    viseme,
                    phoneme.StartTime,
                    phoneme.EndTime,
                    phoneme.Probability
                ));
            }
            
            timeline.Sort();
            return timeline;
        }
        
        private static float ComputeRms(float[] samples, int start, int length)
        {
            float sum = 0;
            int count = 0;
            for (int i = start; i < start + length && i < samples.Length; i++)
            {
                sum += samples[i] * samples[i];
                count++;
            }
            return count > 0 ? MathF.Sqrt(sum / count) : 0;
        }
        
        
        private static float[] Resample(float[] samples, int fromRate, int toRate)
        {
            double ratio = (double)toRate / fromRate;
            int newLength = (int)(samples.Length * ratio);
            var resampled = new float[newLength];
            
            for (int i = 0; i < newLength; i++)
            {
                double srcIndex = i / ratio;
                int srcI = (int)srcIndex;
                double frac = srcIndex - srcI;
                
                if (srcI + 1 < samples.Length)
                    resampled[i] = (float)(samples[srcI] * (1 - frac) + samples[srcI + 1] * frac);
                else if (srcI < samples.Length)
                    resampled[i] = samples[srcI];
            }
            
            return resampled;
        }
    }
}
