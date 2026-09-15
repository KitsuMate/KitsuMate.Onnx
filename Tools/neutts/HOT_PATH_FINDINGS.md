# NeuTTS hot-path optimization — 2026-09-15

## Result

On the current RTX 3070 Laptop GPU, Unity 6000.5.3f1, ONNX Runtime 1.30.0 and
native WebGPU plugin 0.3.0, the saved local configuration produces speech at
approximately real time after warmup. These are complete `RunAsync` measurements,
including prompt processing, token sampling, codec decoding and output copying.
Sessions remain loaded; no audio-response cache is used.

| Input | Audio duration | Previous measured run | Optimized hot runs | Optimized RTF |
| --- | ---: | ---: | ---: | ---: |
| Hello, this is a local voice test. | 2.56 s | 7.264 s | 2.441–2.500 s | 0.95–0.98 |
| The train was late, so we walked along the river and watched the evening lights come on. | 5.04 s | 14.359 s | 4.753–4.826 s | 0.94–0.96 |

The short sample is about 66% faster (roughly 2.9 times the throughput).
The previous short measurement was hot; the previous long measurement was a
single run, so it is not a controlled comparison of repeated long requests.
Repeated exploratory short runs were 2.538–2.566 s, so RTF can still cross 1.0
with normal timing variation. This is not a guarantee for every sentence, voice,
GPU load or power setting. The saved configuration took 6.837 s to load and its
first synthesis took 3.416 s (RTF 1.33); cold start is not real time.

## Changes

`prepare_backbone.py` prepares the existing FP16-storage / FP32-computation export:

- Replace the decomposed RMSNorm formula with ONNX Runtime's fused normalization.
- Replace dynamic calculations of fixed head dimensions with fixed reshape values.
- Fuse causal grouped-query attention, rotary embedding and KV-cache concatenation
  into `com.microsoft.GroupQueryAttention`.
- Precompute the rotary sine/cosine tables for the existing 2048-token context.
- Remove graph nodes and constants made unnecessary by those substitutions.

The runtime interface, full-vocabulary sampling, FP32 arithmetic and existing GPU
cache ownership remain intact. The prepared graph assumes the runtime's batch-one,
contiguous positions and causal attention. `attention_mask` supplies sequence
lengths; arbitrary custom masks or position sequences are not supported by this
specialized export. This matches `NeuTtsEngineRuntime`.

The four-voice profile recorded 1456 GPU attention executions (28 layers × 52
steps), with no CPU attention execution. Per step, only one small shape Gather
and Cast remain on CPU. The cache stays on GPU; the full logits still return to
CPU for sampling. This pass did not introduce another cache download/upload loop.

The native provider and fused operators were tested locally. Browser deployment
was not tested. See the official [native WebGPU provider documentation](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html)
and [contributed operator contracts](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md).

## Experiments not selected

| Experiment | Short hot time |
| --- | ---: |
| Existing weight-only INT4 export | 7.873 s |
| Fused RMSNorm only | 5.715 s |
| RMSNorm and simplified shape/head operations | 5.156 s |
| Selective FP16 matrix computation on that graph | 6.658 s |
| Separate fused rotary embedding | 3.540 s |
| Fused grouped attention | 2.566 s |

INT4 produced a slightly longer clip, so its raw time is not an identical-token
comparison. It did not improve RTF enough to select. FP16 matrix computation was
also slower. The selected model introduces no further lossy weight conversion.

## Validation

- `test_prepare_backbone.py`: three tests passed, including numerical normalization
  equivalence at 12 sequence/magnitude combinations, residual preservation and
  rejection of incompatible normalization/backbone graphs.
- `validate_backbone.py`: four voices, including happy emotion, with 13 reference-
  forced steps each. Compared against the original FP16-storage graph on CPU.
  All top-50 candidate sets matched. Worst logit difference was 0.0000496;
  worst new KV-cache difference was 0.0000113. All outputs were finite.
- Whisper recovered the exact text from six WAVs: all four voices on the short
  sentence, plus short and long benchmark outputs.
- Three seeded short outputs were identical; three seeded long outputs were
  identical. The short output was also byte-identical to the previous good model.
- Mid-run cancellation was observed. Subsequent synthesis succeeded and produced
  the same bytes as before cancellation.
- Actual saved asset loaded and ran in Unity. No C# runtime changes were necessary
  in this pass. The full TTS NUnit package remains unavailable in this project.

## Reproduction and saved configuration

```powershell
python prepare_backbone.py backbone_fp16.onnx backbone_webgpu.onnx
python -m unittest discover -s . -p test_prepare_backbone.py
python validate_backbone.py ARTIFACTS backbone_fp16.onnx backbone_webgpu.onnx --webgpu --device 0 --report parity.json
```

Use a Python environment with ONNX, ONNX Runtime 1.30.0, the WebGPU plugin and
tokenizers. Select the desired adapter index. The validation CLI includes CPU
cache transfers and is for correctness/operator placement, not RTF measurement.

`Assets/Onnx/Tts/NeuTTS/NeuTtsModelSet.asset` now selects
`NeuTtsModelSet Files/onnx/backbone_webgpu.onnx`, alongside the previously repaired
GPU codec. Its local revision is `local-webgpu-fused-attention`.
Original model files remain available. The download configuration and hosted
repository have not been changed; downloading that older configuration replaces
the local model selection.

Raw WAVs, profiles, parity results and timing records are under
`.benchmark-onnx/neutts/webgpu-investigation`: `prepared-parity.json`,
`prepared-asr.json`, `prepared-timing.txt`, and `saved-performance.txt`.
