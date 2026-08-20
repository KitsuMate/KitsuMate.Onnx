#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_TOOL_CACHE:?Set a persistent CI tool cache directory.}"

cache_root="${KITSUMATE_CACHE_ROOT:-$RUNNER_TOOL_CACHE/kitsumate}"
artifact_cache="${KITSUMATE_ARTIFACT_CACHE:-$cache_root/artifacts}"
model_cache="${KITSUMATE_MODEL_CACHE:-$cache_root/models}"
mkdir -p "$artifact_cache" "$model_cache"

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
backend_plugins="$repository_root/Packages/ai.kitsumate.onnx.backend.onnxruntime/Runtime/Plugins"

runtime_platform="${KITSUMATE_RUNTIME_PLATFORM:-}"
if [ -z "$runtime_platform" ]; then
  case "$(uname -s)" in
    Linux*) runtime_platform="Linux" ;;
    Darwin*) runtime_platform="macOS" ;;
    MINGW*|MSYS*|CYGWIN*) runtime_platform="Windows" ;;
    *)
      echo "Unsupported test host; set KITSUMATE_RUNTIME_PLATFORM explicitly." >&2
      exit 1
      ;;
  esac
fi

python3 "$repository_root/Tools/ci/download-onnxruntime.py" \
  --lock "$repository_root/Dependencies/onnxruntime.lock.json" \
  --cache "$artifact_cache" \
  --destination "$backend_plugins" \
  --platform "$runtime_platform"

fixture_lock="$repository_root/Dependencies/fixtures/ci.lock.json"
if [ ! -f "$fixture_lock" ]; then
  echo "Missing fixture lock: $fixture_lock" >&2
  echo "Create and commit the required checksum-pinned CPU CI fixture set before running integration tests." >&2
  exit 1
fi
python3 "$repository_root/Tools/ci/download-fixtures.py" \
  --lock "$fixture_lock" \
  --cache "$model_cache" \
  --repository-root "$repository_root"

echo "Provisioned ONNX Runtime for $runtime_platform and the required CPU CI fixture set from persistent caches."
echo "The workflow invokes GameCI's unity-test-runner after this step."
