using System.Collections.Generic;

namespace KitsuMate.Onnx.LipSync
{
    /// <summary>
    /// Maps IPA phonemes to standard visemes.
    /// Uses a comprehensive mapping based on articulatory phonetics.
    /// </summary>
    public class StandardPhonemeToVisemeMap : IPhonemeToVisemeMap
    {
        private static readonly Dictionary<string, Viseme> PhonemeToViseme = new()
        {
            // Silence
            { "<blank>", Viseme.sil },
            { "<unk>", Viseme.sil },
            { "", Viseme.sil },
            
            // Bilabial: p, b, m -> PP
            { "p", Viseme.PP }, { "b", Viseme.PP }, { "m", Viseme.PP },
            { "pʰ", Viseme.PP }, { "bʱ", Viseme.PP }, { "ɓ", Viseme.PP },
            { "ʙ", Viseme.PP }, { "β", Viseme.PP },
            
            // Labiodental: f, v -> FF
            { "f", Viseme.FF }, { "v", Viseme.FF },
            { "ʋ", Viseme.FF }, { "ⱱ", Viseme.FF },
            
            // Dental: th -> TH
            { "θ", Viseme.TH }, { "ð", Viseme.TH },
            
            // Alveolar plosives: t, d -> DD
            { "t", Viseme.DD }, { "d", Viseme.DD },
            { "tʰ", Viseme.DD }, { "dʱ", Viseme.DD },
            { "ɗ", Viseme.DD }, { "ɖ", Viseme.DD }, { "ʈ", Viseme.DD },
            
            // Velar: k, g, ng -> kk
            { "k", Viseme.kk }, { "g", Viseme.kk }, { "ɡ", Viseme.kk },
            { "ŋ", Viseme.kk }, { "ɣ", Viseme.kk }, { "x", Viseme.kk },
            { "kʰ", Viseme.kk }, { "gʱ", Viseme.kk },
            { "q", Viseme.kk }, { "ɢ", Viseme.kk },
            
            // Postalveolar: ch, j, sh, zh -> CH
            { "tʃ", Viseme.CH }, { "dʒ", Viseme.CH },
            { "ʃ", Viseme.CH }, { "ʒ", Viseme.CH },
            { "tɕ", Viseme.CH }, { "dʑ", Viseme.CH },
            { "ɕ", Viseme.CH }, { "ʑ", Viseme.CH },
            { "c", Viseme.CH }, { "ɟ", Viseme.CH },
            
            // Alveolar fricatives: s, z -> SS
            { "s", Viseme.SS }, { "z", Viseme.SS },
            { "ʂ", Viseme.SS }, { "ʐ", Viseme.SS },
            { "ts", Viseme.SS }, { "dz", Viseme.SS },
            
            // Alveolar nasal: n -> nn
            { "n", Viseme.nn }, { "ɲ", Viseme.nn },
            { "ɳ", Viseme.nn }, { "ɴ", Viseme.nn },
            
            // Rhotic: r -> RR
            { "ɹ", Viseme.RR }, { "r", Viseme.RR },
            { "ɾ", Viseme.RR }, { "ɽ", Viseme.RR },
            { "ʀ", Viseme.RR }, { "ʁ", Viseme.RR },
            
            // Lateral: l
            { "l", Viseme.nn }, { "ɫ", Viseme.nn },
            { "ɭ", Viseme.nn }, { "ʎ", Viseme.nn },
            { "ɬ", Viseme.nn }, { "ɮ", Viseme.nn },
            
            // Glottal
            { "h", Viseme.sil }, { "ɦ", Viseme.sil },
            { "ʔ", Viseme.sil }, { "ʕ", Viseme.sil },
            
            // Open vowels: a -> aa
            { "a", Viseme.aa }, { "ɑ", Viseme.aa },
            { "ɐ", Viseme.aa }, { "æ", Viseme.aa },
            { "ä", Viseme.aa }, { "ɒ", Viseme.aa },
            
            // Mid-front vowels: e -> E
            { "e", Viseme.E }, { "ɛ", Viseme.E },
            { "ɜ", Viseme.E }, { "ɝ", Viseme.E },
            { "ə", Viseme.E }, { "ɚ", Viseme.E },
            
            // High-front vowels: i -> ih
            { "i", Viseme.ih }, { "ɪ", Viseme.ih },
            { "ɨ", Viseme.ih }, { "ʏ", Viseme.ih },
            { "y", Viseme.ih },
            
            // Mid-back vowels: o -> oh
            { "o", Viseme.oh }, { "ɔ", Viseme.oh },
            { "ø", Viseme.oh }, { "œ", Viseme.oh },
            
            // High-back vowels: u -> ou
            { "u", Viseme.ou }, { "ʊ", Viseme.ou },
            { "ɯ", Viseme.ou }, { "ʉ", Viseme.ou },
            
            // Approximants
            { "w", Viseme.ou }, { "ʍ", Viseme.ou },
            { "j", Viseme.ih }, { "ɥ", Viseme.ih },
        };
        
        private static readonly Dictionary<Viseme, List<string>> VisemeToPhonemes = new();
        
        static StandardPhonemeToVisemeMap()
        {
            // Build reverse mapping
            foreach (var kvp in PhonemeToViseme)
            {
                if (!VisemeToPhonemes.TryGetValue(kvp.Value, out var list))
                {
                    list = new List<string>();
                    VisemeToPhonemes[kvp.Value] = list;
                }
                list.Add(kvp.Key);
            }
        }
        
        public Viseme MapPhoneme(string phoneme)
        {
            if (string.IsNullOrEmpty(phoneme))
                return Viseme.sil;
            
            // Try exact match
            if (PhonemeToViseme.TryGetValue(phoneme, out var viseme))
                return viseme;
            
            // Try lowercase
            if (PhonemeToViseme.TryGetValue(phoneme.ToLowerInvariant(), out viseme))
                return viseme;
            
            // Try first character for unknown phonemes
            if (PhonemeToViseme.TryGetValue(phoneme[0].ToString(), out viseme))
                return viseme;
            
            return Viseme.sil;
        }
        
        public IReadOnlyList<string> GetPhonemesForViseme(Viseme viseme)
        {
            return VisemeToPhonemes.TryGetValue(viseme, out var list) 
                ? list 
                : new List<string>();
        }
    }
}
