# KitsuMate ONNX Runtime NVIDIA Providers

Install this package alongside `ai.kitsumate.onnx.backend.onnxruntime` and add
`KITSUMATE_ORT_NVIDIA` to a Windows x64 or Linux x64 Build Profile. Automatic
selection then tries TensorRT-RTX, CUDA, the platform default, and CPU.

`KITSUMATE_ORT_CUDA` remains a deprecated CUDA-only alias. The legacy pair
`KITSUMATE_ORT_CUDA` + `KITSUMATE_ORT_TENSORRT` enables NVIDIA automatic behavior
and emits a migration warning.

TODO: after the provider ABI and payload layout have stabilized, extract the CUDA
and TensorRT-RTX closures into independently versioned provider payload packages
without changing the public registry contract.
