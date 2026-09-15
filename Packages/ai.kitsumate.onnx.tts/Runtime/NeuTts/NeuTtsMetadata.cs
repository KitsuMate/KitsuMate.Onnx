using System;
using System.Linq;

namespace KitsuMate.Onnx.Tts.NeuTts
{
    [Serializable]
    internal sealed class NeuTtsMetadata
    {
        public string family;
        public int sampleRate, maxContext, layers, kvHeads, headDim;
        public int speechStart, speechEnd, textStart, textEnd;
        public int[] speechTokenIds;
        public Speaker[] speakers;
        public Emotion[] emotions;
        [Serializable] public sealed class Speaker { public string name, text; public int[] codes; }
        [Serializable] public sealed class Emotion { public string name; public int token; }

        public void Validate()
        {
            if (family != "neutts-2e" || sampleRate != 24000 || maxContext < 51 ||
                layers <= 0 || kvHeads <= 0 || headDim <= 0 || speechTokenIds == null ||
                speechTokenIds.Length == 0 || speechTokenIds.Any(id => id < 0) ||
                speechTokenIds.Distinct().Count() != speechTokenIds.Length ||
                speakers == null || speakers.Length != 4 || emotions == null || emotions.Length != 7)
                throw new ArgumentException("Invalid or unsupported NeuTTS model metadata.");
            foreach (var speaker in speakers)
                if (string.IsNullOrWhiteSpace(speaker.name) || string.IsNullOrWhiteSpace(speaker.text) ||
                    speaker.codes == null || speaker.codes.Length == 0 ||
                    speaker.codes.Any(code => code < 0 || code >= speechTokenIds.Length))
                    throw new ArgumentException("Invalid NeuTTS speaker reference.");
        }
    }
}
