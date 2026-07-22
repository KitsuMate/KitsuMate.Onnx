# KitsuMate ONNX ASR

Automatic Speech Recognition package with Whisper and Wav2Vec implementations.

## Features

- **WhisperEngine**: OpenAI Whisper-based transcription
  - Multi-language support with language detection
  - Word-level timestamps
  - Force alignment for subtitle generation
  - Merged or split cached decoders

- **SentisWhisperEngine**: Whisper Tiny and Base on Sentis 2.6.1
  - Available when `ai.kitsumate.onnx.backend.unity-inference` is installed
  - Uses split FP32 decoder models

- **Wav2VecEngine**: Facebook Wav2Vec2-based transcription
  - CTC decoding
  - Lightweight alternative to Whisper

## Installation

Add to your `manifest.json`:

```json
{
  "dependencies": {
    "ai.kitsumate.onnx.asr": "file:../Packages/ai.kitsumate.onnx/asr"
  }
}
```

## Quick Start

1. Import Whisper ONNX models (encoder, decoder, mel processor)
2. Create a `WhisperModelSet` asset and assign models
3. Create a `WhisperEngine` asset and assign the model set
4. Add `SpeechToText` component to a GameObject
5. Assign the engine and configure audio input

```csharp
// Programmatic usage
var engine = Resources.Load<WhisperEngine>("MyWhisperEngine");
var result = engine.Run(audioClip);
Debug.Log(result.Text);
```

## Model Sets

### WhisperModelSet
- Encoder model
- Decoder model  
- Mel processor model
- Tokenizer JSON

### Wav2VecModelSet
- Processor model
- Vocabulary file

## Components

### SpeechToText
MonoBehaviour for real-time or clip-based transcription.

```csharp
public class SpeechToText : MonoBehaviour
{
    [SerializeField] private AsrEngine _engine;
    public UnityEvent<TranscriptionResult> OnTranscription;
}
```

## License

MIT
