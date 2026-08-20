# Third-party notices

This package redistributes checksum-pinned artifacts from:

- ONNX Runtime 1.25.1 under the MIT license and its bundled third-party notices.
- NVIDIA CUDA 12.9 and cuDNN 9 runtime redistributables under NVIDIA's applicable SDK/runtime terms.
- TensorRT-RTX EP ABI 0.3.0/cu12 and TensorRT-RTX 1.5 runtime under the license and acknowledgements included in the upstream release archive.

Release hydration must retain the complete upstream license and acknowledgement files alongside the published UPM archive. See `Dependencies/onnxruntime-nvidia.lock.json` for exact origins and checksums.

TODO(provider package split): move each upstream notice with its payload when CUDA and TensorRT-RTX become independently versioned packages.
