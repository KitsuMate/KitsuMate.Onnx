# NeuTTS runtime comparison — 2026-09-15

## Local PyTorch comparison

The same locally exported NeuTTS-2E source checkpoint and two benchmark prompts
were tested on the RTX 3070 Laptop GPU. PyTorch used the upstream-style
`AutoModelForCausalLM` BF16 load and `generate` path, CUDA SDPA, cache enabled,
temperature 1, top-k 50, top-p 1, seed 42, and a 50-token minimum. This was
uncompiled PyTorch 2.5.1+cu124 with Transformers 5.9.0 and four CPU threads.
CUDA synchronization bracketed timing. Model loading was excluded.

PyTorch measured 15.1–16.4 generated tokens/s on hot runs, including prompt processing
and sampling but excluding audio decoding. The optimized ONNX/WebGPU implementation
measured about 52–53 speech tokens/s including the codec. Thus the current local
ONNX path achieved roughly 3.2–3.5 times this PyTorch baseline's token rate, despite
including more work. This does not establish a limit on optimized PyTorch speed:
`torch.compile`, other versions, and specialized generation loops were not tested.

The seed does not synchronize random sampling between C# and PyTorch. The short
PyTorch take generated 170 speech tokens versus 129 in the ONNX take; the long
take generated 280 versus 253. Compare token throughput, not raw sentence times.
PyTorch's reported generated-token count includes one end token per request.
This pass did not decode or transcribe the PyTorch output.

Raw results and benchmark script are in `.benchmark-onnx/neutts/`:
`webgpu-investigation/pytorch-comparison.json` and `benchmark_pytorch.py`.
The current ONNX measurements are documented in [HOT_PATH_FINDINGS.md](HOT_PATH_FINDINGS.md).

## Published GGUF and GPU throughput

The [upstream README](https://github.com/neuphonic/neutts#throughput-benchmarking)
publishes the following figures for Air and Nano, not NeuTTS-2E:

| Device / runtime | Air | Nano |
| --- | ---: | ---: |
| Galaxy A25 CPU, GGUF Q4_0 / llama.cpp | 20 tokens/s | 45 tokens/s |
| Ryzen 9 HX 370 CPU, GGUF Q4_0 / llama.cpp | 119 tokens/s | 221 tokens/s |
| iMac M4 CPU, GGUF Q4_0 / llama.cpp | 111 tokens/s | 195 tokens/s |
| RTX 4090, vLLM throughput benchmark | 16,194 tokens/s | 19,268 tokens/s |

The CPU setup uses 500 prompt tokens and 250 generated tokens, with 14/16
prefill/decode threads on the desktop CPUs and six on the phone. The codec is
excluded. The vLLM figures are throughput benchmark results, not a comparable
single-request PyTorch latency measurement on this laptop.

These figures establish that optimized GGUF can exceed real-time backbone speed
on capable CPUs. They do not establish whether GGUF 2E is faster than this ONNX
implementation on the RTX 3070 Laptop GPU. A local GGUF 2E runtime/model was not
installed, and no same-device GGUF run was performed. The
[2E model card](https://huggingface.co/neuphonic/neutts-2e) makes real-time CPU
claims but supplies no numerical PyTorch-versus-GGUF comparison.

