# KitsuMate ONNX TTS

Provider-neutral ONNX text-to-speech for Unity. Chatterbox remains supported, and OmniVoice is
available as a separate `TtsEngine` with automatic voices, instructed voice design, and voice
cloning.

## NeuTTS-2E

`NeuTtsEngine` supports English emotional synthesis with `emily`, `paul`, `sophie`, and `steven`.
Create a NeuTTS Engine asset, assign an ONNX backend, and use **Create Model Set and Download**
to select a backbone from [`KitsuMate/neutts-2e-onnx`](https://huggingface.co/KitsuMate/neutts-2e-onnx).
FP32, FP16 weight storage, dynamic INT8, and compact INT4 share the FP32 NeuCodec decoder. The downloader includes
the tokenizer, speaker references, metadata, licenses, and external-data companions.

```csharp
var request = new TtsRequest("I can't believe it's finally here!")
{
    MaxNewTokens = 700,
    NeuTts = new KitsuMate.Onnx.Tts.NeuTts.NeuTtsGenerationConfig
    {
        Speaker = "emily", Emotion = "happy", UseSeed = true, Seed = 42
    }
};
TtsResult result = await runtime.RunAsync(request);
```

Emotions are `angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, and `surprised`.
Null NeuTts options use engine defaults. Requests return a complete mono 24 kHz waveform.
Optional `TopP` and `MinP` sampling controls default to 1 and 0. Use 0.95 and 0.05
to try the community sampler while keeping speaker conditioning unchanged.
The repository also provides FP16 weight storage with FP32 computation and a dynamic INT8
CPU backbone. FP16 halves the backbone download size; it does not imply FP16 computation
or reduced runtime memory. All profiles use the same FP32 codec and tensor interfaces.
NeuTTS sessions use up to four CPU threads, matching the local comparison settings.
The backbone follows the backend provider policy and retains KV-cache tensors on the
device when IO binding is available; only logits return to CPU. Loaded sessions are reused
across requests. Small top-k sampling avoids sorting the full vocabulary.
**Use CPU Codec** in the model set defaults to enabled because WebGPU corrupts audio
with the original codec's stride-480 ConvTranspose. For GPU decoding, regenerate the codec
with `Tools/neutts/prepare_codec.py`, assign that graph, and disable **Use CPU Codec**.
The prepared graph uses an equivalent overlap Conv without changing the learned weights.
It also uses tensor Gather instead of CPU-only sequence operations inside the codec.
For WebGPU, prefer the FP32 or FP16-storage backbone. Dynamic INT8 matrix operations
fall back to CPU and introduce many transfers between CPU and GPU within each token step.
The 2048-token context includes text and reference speech; oversized prompts are rejected.
MaxNewTokens caps generation and may truncate speech if too small (roughly 50 tokens per second).
Exaggeration, RepetitionPenalty, Speed, and DurationSeconds are not NeuTTS controls.
Custom reference audio, voice design, streaming, Air and Nano are not supported in this release.
The runtime returns raw codec floats, which can exceed +/-1; limit peaks before integer PCM conversion.
It does not apply upstream's optional Perth watermark.
Literal digits and times are unreliable in the source model; spell numbers out in words before synthesis.

The backbone retains the NeuTTS Open License v1.0, including its commercial-use restriction;
the codec and reference assets retain Apache-2.0. See the model repository for licenses and validation
limits, and `Tools/neutts` in the source repository for reproducible conversion scripts.

## OmniVoice artifacts

- **CPU compact INT4** uses the bidirectional merged backbone with 4-bit symmetric weight-only
  quantization in 128-weight blocks and the standard FP32 Higgs codec. It is derived from the
  validated Portable FP32 graph rather than the incompatible causal split export.
- **Portable FP32** uses the merged, standard-ONNX opset-17 backbone and FP32 codec. Provider order
  belongs to `OnnxRuntimeBackend`; the engine never selects CUDA, WebGPU, TensorRT, or OpenVINO
  itself.

The upstream split INT4 and CPU-FP16 exports were rejected because their language decoder is
causal while OmniVoice masked diffusion requires bidirectional attention. The curated merged INT4
profile preserves bidirectional behavior and passes Whisper checks for auto, design, and clone.

These artifacts are available in
[`KitsuMate/omnivoice-onnx`](https://huggingface.co/KitsuMate/omnivoice-onnx). The downloader also
recognizes the split [`onnx-community/OmniVoice-Onnx`](https://huggingface.co/onnx-community/OmniVoice-Onnx)
layout and the merged [`gluschenko/omnivoice-onnx`](https://huggingface.co/gluschenko/omnivoice-onnx)
layout. A merged backbone repository is incomplete without a compatible tokenizer and all four
Higgs codec graphs.

Each model set saves independent artifact selections for its submodels in the shared downloader. Codec precision is read from graph metadata. Model files and companions use the installation root configured in `OnnxSettings`.

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
