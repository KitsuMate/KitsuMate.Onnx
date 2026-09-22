# KitsuMate ONNX TTS

Provider-neutral ONNX text-to-speech for Unity. Chatterbox remains supported, and OmniVoice is
available as a separate `TtsEngine` with automatic voices, instructed voice design, and voice
cloning.

## Chatterbox Multilingual V3

V3 requires an export of `t3_mtl23ls_v3.safetensors` with its matching text vocabulary
and real voice-encoder conditioning. Renaming an older multilingual ONNX model is
not sufficient. The V3 embedding graph includes a `text_conditioning` input; the
runtime uses it to recognize this graph interface and enable classifier-free guidance.

`ChatterboxGenerationConfig` controls temperature, top-p, min-p, guidance and the
sampling seed. V3 defaults are 0.8, 0.95, 0.05, 0.5 and 42 respectively. Guidance
uses conditional and unconditional batches. Setting guidance to zero uses one batch;
its quality and speed must be measured for the intended language and voice before
selecting it as a default. A zero temperature selects greedy decoding. The split
decoder also uses this seed for its initial flow noise.

Chatterbox downloads require a tokenizer but may omit a bundled default voice.
Supply `TtsRequest.VoiceReference` or assign a default voice on the model set;
requests without either fail before inference.

V3 preserves text case and uses NFKD normalization. The converted tokenizer supplies
the embedding graph's control tokens; the runtime inserts `[SPACE]` before encoding.
The bundled tokenizer includes the Whitespace pre-tokenizer correction, tested against
Python tokenizers reference IDs. Optional model companions provide longest-match Japanese
kanji readings, an unambiguous Russian wordform stress table, and maximum-match Chinese
word segmentation before Cangjie conversion. Unknown readings and words remain unchanged;
the dictionary segmenter is deterministic but not bit-identical to pkuseg's CRF.
Use a prepared, clean reference recording; the exported default voice is prepared by
the upstream six-second VAD and fade procedure, then normalized below +/-1 for Unity
PCM import. Runtime reference preprocessing does
not currently reproduce that VAD procedure. Generated audio has no Perth watermark.

The local conversion is pinned to model revision
`5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` and V3 demo revision
`b21d9d062b4eda102f21975333276919935d2060`, using `s3gen.pt` for the decoder.
Conversion and benchmark tools belong in the ignored `.agent-tools/chatterbox-v3`
directory. Apply the pinned onnxslim 0.1.68 pass after decoder export; the selected
portable decoder has about 41,000 nodes instead of 71,000. The existing ONNX Runtime
backend is sufficient; this integration does not require a new native runtime fork.

The validated local configuration uses FP32 weights, guidance 0.5 and WebGPU with
CPU initialization fallback. Four short/long English and Polish samples passed
exact-word transcription checks in Unity on an RTX 3070 Laptop GPU with ONNX Runtime
1.30.0. Total load time was about 199 seconds. Subsequent Polish short, English long
and Polish long requests took 7.2, 15.3 and 20.7 seconds respectively. These results
do not establish parity for every language, voice or GPU, or real-time synthesis.
Dynamic INT8 failed to stop on a long Polish passage. INT4 introduced small word
regressions, so neither is selected. A saved CPU ORT decoder failed WebGPU buffer
placement; use portable ONNX for this configuration.

Full FP16 computation is not a validated V3 profile on native WebGPU 0.3.0.
Tests found both language-model precision regressions and incorrect FP16
ConvTranspose coordinates in waveform decoding. Keeping the decoder's five
transposed convolutions in FP32 fixes the isolated kernel checks while retaining
FP16 stored weights; it does not resolve the separate language-model regressions.
FP16 weight storage with FP32 arithmetic roughly halves model disk size but did
not improve synthesis speed in the local comparison. Keep FP32 as the default.
Graph conversion and standalone Python WebGPU regression tools remain in the
owning repository's ignored `.agent-tools/chatterbox-v3` directory.

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
with the original codec's stride-480 ConvTranspose.
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
the codec and reference assets retain Apache-2.0. See the model repository for licenses and
validation limits.

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
