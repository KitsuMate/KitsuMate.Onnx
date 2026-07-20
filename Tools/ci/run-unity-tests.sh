#!/usr/bin/env bash
set -euo pipefail

: "${KITSUMATE_ARTIFACT_CACHE:?Set a writable cache directory for checksum-pinned artifacts.}"
: "${KITSUMATE_UNITY_COMMAND:?Set the reviewed GameCI Unity test command on the protected runner.}"

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
backend_plugins="$repository_root/Packages/ai.kitsumate.onnx.backend.onnxruntime/Runtime/Plugins"

python3 "$repository_root/Tools/ci/hydrate-onnxruntime.py" \
  --lock "$repository_root/Dependencies/onnxruntime.lock.json" \
  --cache "$KITSUMATE_ARTIFACT_CACHE" \
  --destination "$backend_plugins"

mkdir -p "$repository_root/TestResults"
cd "$repository_root"
eval "$KITSUMATE_UNITY_COMMAND"
