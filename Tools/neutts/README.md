# NeuTTS-2E conversion

Run these commands from the KitsuMate.Onnx root. Keep model artifacts outside Git.
Use an isolated Python environment with `requirements.txt`; no Python is required by Unity inference.

1. Authenticate with `hf auth login` and obtain access to `neuphonic/neutts-2e` and
   `neuphonic/neucodec-onnx-decoder`.
2. Download NeuTTS-2E revision `e24ca17d47b8cdd1cdab45792013b5a81547d1a5` into `SOURCE`.
3. Download the codec revision `ac01424f74ea25b245132252168e2191b3d859fa`.
4. Check out `https://github.com/neuphonic/neutts` revision
   `ac69851f28fc63a487917e7c2e27f0d75c759cba` into `UPSTREAM`.
5. Export and validate:

```sh
python Tools/neutts/export.py --source SOURCE --upstream UPSTREAM --output ARTIFACTS
python Tools/neutts/quantize.py ARTIFACTS
python Tools/neutts/precision.py ARTIFACTS --profile fp16
python Tools/neutts/precision.py ARTIFACTS --profile int8
python Tools/neutts/prepare_codec.py CODEC/model.onnx ARTIFACTS/onnx/codec_decoder.onnx
python Tools/neutts/test_prepare_codec.py
python Tools/neutts/validate_codec.py CODEC/model.onnx ARTIFACTS/onnx/codec_decoder.onnx ARTIFACTS/neutts.json --report ARTIFACTS/codec-parity.json
python Tools/neutts/reference_parity.py --source SOURCE --upstream UPSTREAM --artifacts ARTIFACTS
python Tools/neutts/validate.py ARTIFACTS --profile fp32
python Tools/neutts/validate.py ARTIFACTS --profile int4
python Tools/neutts/validate.py ARTIFACTS --profile fp16
python Tools/neutts/validate.py ARTIFACTS --profile int8
python Tools/neutts/compare_profiles.py ARTIFACTS
```

`gpu_validate.py` additionally requires the matching `onnxruntime-gpu==1.24.4` distribution,
CUDA/cuDNN libraries, and an NVIDIA GPU. Run it in a separate environment from CPU ORT.
`transcribe.py` requires `faster-whisper==1.2.1`, `ctranslate2==4.7.1`, and `av==16.1.0`;
pass a local Whisper-base checkpoint with `--model`.

Copy the Apache-2.0 license text to `ARTIFACTS/CODEC_LICENSE`, then run
`prepare_repository.py ARTIFACTS`. It requires completed matrix and reference validation
and writes graph schemas, attribution, the model card, and checksums. Only upload files listed
in `checksums.json`, plus the checksum file itself. Profiling traces and source weights are not release files.

For Unity tests, install/test-enable `ai.kitsumate.onnx.tts.tests` and run `NeuTtsTests`
in PlayMode. Set `KITSUMATE_NEUTTS_ARTIFACTS` to the prepared artifact directory before starting Unity,
or use the development default `<project>/.benchmark-onnx/neutts/artifacts`.
Integration cases test CPU FP32, CPU INT4 and WebGPU FP32 with provider fallback disabled.

The prepared codec replaces the iSTFT stride-480 ConvTranspose with an equivalent
stride-one Conv and sample rearrangement. It also replaces unbind sequence operators
with tensor Gather, removing CPU/GPU round trips inside the codec's attention layers.
Native WebGPU's ConvTranspose shader can
reject valid sample positions due to its floating-point division/integer test.
Keep the original source codec for CPU parity comparisons. Assign the prepared graph
to the NeuTTS model set and disable **Use CPU Codec** to use the backend's provider
for the codec. Keep that option enabled with the original downloaded graph.
The backbone keeps KV-cache tensors on the device through the existing IO-binding API;
only logits return to CPU for sampling.
Use the FP32 or FP16-storage backbone for WebGPU. The dynamic INT8 graph's CPU-only
integer matrix operations introduce hundreds of internal transfers per token step.
To check native WebGPU parity, run `validate_codec.py` with `--webgpu` in an environment
with compatible `onnxruntime` and `onnxruntime-ep-webgpu` packages (validated locally
with 1.30.0 and 0.3.0). `--device` selects an index among the WebGPU devices.

FP16 stores matrix weights at half precision and computes in FP32. Full FP16 compute
showed excessive drift, so it is not shipped. Dynamic INT8 uses QInt8 MatMul and Gather.
The shared float32 cache/mask/logit interface stays compatible with existing Unity runtimes.
Run `transcribe.py ARTIFACTS --model WHISPER --profiles fp32 fp16 int8 int4` before publication.
Optional sampling flags `--top-p 0.95 --min-p 0.05` match the community sampler in
Danny-Dasilva/neutts-2e-onnx revision `e3ea12b860d75acf6903ddd4fbb7c72c0ed4eae3`.
Defaults remain top-p 1 and min-p 0, matching upstream PyTorch behavior.

Keep the original NeuTTS license attached to derived backbones. See each generated repository's
ATTRIBUTION.md for source revisions and conversion notices. No watermark is applied by this runtime.
