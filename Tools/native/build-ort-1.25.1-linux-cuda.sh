#!/usr/bin/env bash
set -euo pipefail

source_root="${1:?ONNX Runtime source root is required}"
output_root="${2:?output directory is required}"
expected_commit="8a77e459420f58fb946fd9067285cfa719f10bdd"
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$expected_commit" ]]

mkdir -p "$output_root"
# TODO(provider package split): promote this CUDA 12.9/cuDNN 9 toolchain to an
# independently versioned provider build image after the plug-in ABI stabilizes.
"$source_root/build.sh" \
  --config Release --build_shared_lib --parallel --skip_tests --update --build \
  --use_cuda --cuda_home "${CUDA_HOME:-/usr/local/cuda-12.9}" \
  --cudnn_home "${CUDNN_HOME:-${CUDA_HOME:-/usr/local/cuda-12.9}}" \
  --cmake_extra_defines onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON

cp "$source_root/build/Linux/Release/libonnxruntime_providers_cuda.so" "$output_root/"
sha256sum "$output_root/libonnxruntime_providers_cuda.so"
