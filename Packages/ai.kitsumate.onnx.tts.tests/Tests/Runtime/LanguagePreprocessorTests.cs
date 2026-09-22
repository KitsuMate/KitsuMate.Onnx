using KitsuMate.Onnx.Tts.Chatterbox;
using NUnit.Framework;
using System.Text;

namespace KitsuMate.Onnx.Tts.Tests
{
    public sealed class LanguagePreprocessorTests
    {
        [Test]
        public void ProcessV3_JapaneseCompoundsAndOkurigana_UsesLongestUnambiguousReadings()
        {
            const string readings =
                "日\tニチ\n" +
                "日本\tニホン\n" +
                "日本語\tニホンゴ\n" +
                "食べる\tタベル\n";
            var preprocessor = new LanguagePreprocessor(japaneseReadingData: readings);

            Assert.That(preprocessor.ProcessV3("日本語を食べる", "JA"),
                Is.EqualTo("にほんごをたべる".Normalize(NormalizationForm.FormKD)));
        }

        [Test]
        public void ProcessV3_JapaneseAmbiguousReading_LeavesKanjiUnchanged()
        {
            const string readings = "生\tセイ\n生\tショウ\n";
            var preprocessor = new LanguagePreprocessor(japaneseReadingData: readings);

            Assert.That(preprocessor.ProcessV3("生", "ja"), Is.EqualTo("生"));
        }

        [Test]
        public void ProcessV3_RussianWordforms_InsertsKnownStressAndSkipsHomographs()
        {
            const string stresses = "молоко\t5\nслова\t2\nслова\t4\n";
            var preprocessor = new LanguagePreprocessor(russianStressData: stresses);

            Assert.That(preprocessor.ProcessV3("Молоко слова неизвестно", "RU"),
                Is.EqualTo("Молоко́ слова неизвестно"));
        }

        [Test]
        public void ProcessV3_AlreadyStressedRussianWord_DoesNotDuplicateMark()
        {
            var preprocessor = new LanguagePreprocessor(russianStressData: "молоко\t5\n");

            Assert.That(preprocessor.ProcessV3("молоко́", "ru"), Is.EqualTo("молоко́"));
        }

        [Test]
        public void ProcessV3_ChineseText_UsesMaximumMatchingBeforeCangjieConversion()
        {
            const string cangjie = "{\"今\":\"a\",\"天\":\"b\",\"气\":\"c\",\"真\":\"d\",\"好\":\"e\"}";
            const string words = "今天 1000\n天气 900\n真好 800\n";
            var preprocessor = new LanguagePreprocessor(cangjie, chineseWordData: words);

            Assert.That(preprocessor.ProcessV3("今天天气真好", "zh"), Is.EqualTo(
                "[cj_a][cj_.][cj_b][cj_.] [cj_b][cj_.][cj_c][cj_.] [cj_d][cj_.][cj_e][cj_.]"));
        }

        [TestCase("{\"𠀀\":\"f\"}")]
        [TestCase("{\"\\ud840\\udc00\":\"f\"}")]
        public void ProcessV3_NonBmpCangjieKey_ConvertsSingleCodePoint(string cangjie)
        {
            var preprocessor = new LanguagePreprocessor(cangjie);

            Assert.That(preprocessor.ProcessV3("𠀀", "zh"), Is.EqualTo("[cj_f][cj_.]"));
        }

        [Test]
        public void ProcessV3_KoreanSyllable_DecomposesJamo()
        {
            Assert.That(new LanguagePreprocessor().ProcessV3("한", "ko"), Is.EqualTo("한"));
        }
    }
}
