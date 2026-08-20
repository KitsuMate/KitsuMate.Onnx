# KitsuMate ONNX TTS

Provider-neutral ONNX text-to-speech for Unity. Chatterbox remains supported, and OmniVoice is
available as a separate `TtsEngine` with automatic voices, instructed voice design, and voice
cloning.

## OmniVoice profiles

- **CPU compact INT4** uses the bidirectional merged backbone with 4-bit symmetric weight-only
  quantization in 128-weight blocks and the standard FP32 Higgs codec. It is derived from the
  validated Portable FP32 graph rather than the incompatible causal split export.
- **Portable FP32** uses the merged, standard-ONNX opset-17 backbone and FP32 codec. Provider order
  belongs to `OnnxRuntimeBackend`; the engine never selects CUDA, DirectML, TensorRT, or OpenVINO
  itself.

The upstream split INT4 and CPU-FP16 exports were rejected because their language decoder is
causal while OmniVoice masked diffusion requires bidirectional attention. The curated merged INT4
profile preserves bidirectional behavior and passes Whisper checks for auto, design, and clone.

Both profiles are curated in
[`KitsuMate/omnivoice-onnx`](https://huggingface.co/KitsuMate/omnivoice-onnx). The downloader also
recognizes the split [`onnx-community/OmniVoice-Onnx`](https://huggingface.co/onnx-community/OmniVoice-Onnx)
layout and the merged [`gluschenko/omnivoice-onnx`](https://huggingface.co/gluschenko/omnivoice-onnx)
layout. A merged backbone repository is incomplete without a compatible tokenizer and all four
Higgs codec graphs.

## Requests

Set `LanguageId`, `VoiceInstruction`, `Speed`, `DurationSeconds`, and `OmniVoice` decoding options
on `TtsRequest`. Inline non-verbal tags such as `[laughter]`, CMU pronunciation sequences such as
`[B EY1 S]`, and pinyin controls are preserved in the model prompt. Voice cloning additionally
requires both `VoiceReference` and the matching `VoiceReferenceText`; v1 does not run ASR
automatically.

```csharp
var request = new TtsRequest("The [B EY1 S] player finished the song [laughter].")
{
    LanguageId = "en",
    VoiceInstruction = "female, young adult, high pitch, british accent",
    Speed = 1f,
    DurationSeconds = 1.8f
};
TtsResult result = await runtime.RunAsync(request);
```

For a required CUDA smoke test, configure the active Unity Build Profile with
`KITSUMATE_ORT_CUDA`, package the matching CUDA runtime dependencies, and call
`SetProviderOrder(OnnxExecutionProvider.Cuda)`. Omitting CPU makes provider initialization failure
fatal. Other providers use the same model and generation code when their native payloads are
available.

## Attribution

OmniVoice was created by the OmniVoice authors and is distributed under Apache-2.0. The curated
ONNX repository preserves its license and citation, credits the original authors, and records the
two ONNX conversion sources and their exact revisions in `ATTRIBUTION.md` and
`omnivoice-manifest.json`.
