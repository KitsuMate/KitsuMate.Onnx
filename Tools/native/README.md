# ONNX Runtime native builds

The default Windows DirectML core and NVIDIA CUDA plug-in are built from the
exact ONNX Runtime v1.25.1 commit `8a77e459420f58fb946fd9067285cfa719f10bdd`.
CUDA builds require CUDA 12.9 and cuDNN 9. Release artifacts are promoted only
after PE/ELF dependency inspection and their output hashes are added to the
corresponding artifact lock.

TensorRT-RTX is not built here. The NVIDIA package consumes checksum-pinned
standalone EP ABI v0.3.0/cu12 release artifacts.
