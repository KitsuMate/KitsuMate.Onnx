# NeuTTS WebGPU investigation

For the subsequent optimization to approximately RTF 1 on the same setup, see
[HOT_PATH_FINDINGS.md](HOT_PATH_FINDINGS.md). The measurements below document the
earlier correctness fixes and initial performance baseline.

Verified 2026-09-15 on Unity 6000.5.3f1, ONNX Runtime 1.30.0,
WebGPU plugin 0.3.0, NVIDIA GeForce RTX 3070 Laptop GPU.

## Audio corruption: the codec's stride-480 ConvTranspose

The original decoder's intermediate tensors agree between CPU and WebGPU until
`node_ConvTranspose_2593`. Its input has shape `[1,1922,402]`, its inverse Fourier
basis has shape `[1922,1,1920]`, and its stride is 480. At this node the relative
waveform error jumps from approximately 0.000004 to 0.876. Disabling graph
optimizations does not fix it. Standalone Python reproduces the Unity failure.

A single-channel, all-ones ConvTranspose reproduces the problem without any model
weights: input `[1,1,402]`, kernel `[1,1,1920]`, stride 480, no padding. WebGPU gets
164,640 of 194,400 output samples wrong. Strides 1, 2, 3 and 512 pass; 7, 120, 240
and 480 fail on this machine.

The [upstream shader](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webgpu/nn/conv_backprop.cc)
calculates source coordinates with floating-point division and rejects fractional
results. Reciprocal rounding makes mathematically integral positions slightly
fractional: for example, FP32 `1440 * (1 / 480)` is about 3.00000024. Simulating
this calculation predicts every erroneous sample in the minimal stride-480 test
(zero disagreements out of 194,400 samples). This drops valid overlap contributions
and explains the periodic damage to speech. It is not a Unity audio playback issue.

`prepare_codec.py` replaces the operation with an equivalent stride-one Conv,
Transpose and Reshape. For hop H, sample offset r and overlap q, the original is:

    y[t*H+r] = sum(q,c) x[c,t-q] * basis[c,0,q*H+r]

Packing r into output channels and reversing q produces the stride-one Conv.
This preserves learned weights, sample count, dynamic sequence length and batch
dimensions. No native runtime rebuild is needed.

## CPU/GPU transfers: separate performance and readback issues

1. **External KV-cache copies.** NeuTTS called the CPU-array `Run` API for every
   backbone step. All 56 cache outputs were downloaded and uploaded again on the
   next step. For the short test's 444-token prefix and 130 backbone calls, this
   represents approximately 15.1 GB of avoidable cache transfers. The runtime now
   uses `IOnnxDeviceSession.RunOnDevice`: cache tensors remain on device and only
   logits return for sampling. CPU-only sessions and other backends remain supported.

2. **INT8 graph partitioning.** The selected dynamic INT8 backbone executes its
   integer matrix operations on CPU. A three-step native profile recorded 933
   `MemcpyFromHost` and 339 `MemcpyToHost` nodes: 424 internal copies per step.
   FP16-storage with FP32 arithmetic executes matrix operations on WebGPU; the
   equivalent profile records three `MemcpyFromHost` nodes, one per step. Small
   shape/control operations still run on CPU. A WebGPU session label alone does
   not mean every graph operation runs on GPU.

3. **Codec sequence operators.** Twelve `SplitToSequence` operations and 36
   `SequenceAt` operations caused 12 downloads and 36 uploads per decode.
   `prepare_codec.py` replaces this unbind/index pattern with equivalent tensor
   Gather operations. The prepared codec profile has none of those copy nodes.

4. **Device readback.** `OrtDeviceTensor.CopyToCpu` accessed a tensor span directly,
   which does not itself download GPU memory. It now checks memory placement and
   uses synchronous `OrtEnv.CopyTensors` before reading device allocations. This
   matters for explicit GPU readback and provider fallback. It was not the cause
   of the original NeuCodec corruption because that path already requested CPU outputs.
   A live Editor test read a `WebGPU_Buf` allocation and reused it across five GPU
   steps, checking every result. See the [ORT tensor API](https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.OrtValue.html).

## Local timing and validation

Input: `Hello, this is a local voice test.` Seed 42; 129 speech codes for the
WebGPU comparisons. Times exclude session loading.

| Configuration | First measured run | Repeated run |
| --- | ---: | ---: |
| Original INT8, original WebGPU codec, CPU-array cache | 42.525 s | 44.007 s |
| Earlier CPU codec fix and sampling/copy optimization | 32.390 s | 32.460 s |
| Repaired WebGPU codec and device cache, INT8 backbone | 20.881 s | 20.034 s |
| FP32 backbone and device cache | 8.163 s | 7.098 s |
| Final saved local FP16-storage configuration | 8.298 s | 7.264 s |

The final session load was 9.376 seconds. Loaded sessions are reused. The local
model set selects the FP16-storage backbone, the regenerated codec, and disables
Use CPU Codec. Original downloaded files remain intact; downloads restore the safe
CPU codec option. The hosted repository has not been republished.

Validation:

- Three Python regression tests cover convolution equivalence over nine hop/batch/
  length combinations, sequence-index equivalence, and rejection of unrelated graphs.
- `validate_codec.py` compared all four speaker references at three lengths on CPU
  and WebGPU: 24 comparisons. Worst CPU peak error was below 0.0000005. Worst GPU
  peak error was 0.000398 and relative L2 error 0.000101 (about -80 dB).
  GPU validation uses both peak error and signal-relative error rather than relative
  error at individual audio zero crossings. This is numerical parity within the
  documented tolerance, not bitwise equality with CPU.
- Local Whisper recovered the exact short input and the longer sentence:
  `The train was late, so we walked along the river and watched the evening lights come on.`
- Repeated seeded final WAVs were byte-identical, including after mid-run cancellation.
- Runtime changes compiled in Unity and GPU readback/reuse passed direct Editor checks.
  The TTS NUnit package is not installed in this project, so its full suite was not run.

Raw profiles, the minimal reproducer, WAVs, and JSON results are in the project's
`.benchmark-onnx/neutts/webgpu-investigation` directory. Prepared local model graphs
are under `Assets/Onnx/Tts/NeuTTS/NeuTtsModelSet Files/onnx`.
