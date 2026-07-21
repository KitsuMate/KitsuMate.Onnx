#!/usr/bin/env bash
set -euo pipefail

: "${KITSUMATE_ARTIFACT_CACHE:?Set a writable cache directory for checksum-pinned artifacts.}"

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
backend_plugins="$repository_root/Packages/ai.kitsumate.onnx.backend.onnxruntime/Runtime/Plugins"

python3 "$repository_root/Tools/ci/hydrate-onnxruntime.py" \
  --lock "$repository_root/Dependencies/onnxruntime.lock.json" \
  --cache "$KITSUMATE_ARTIFACT_CACHE" \
  --destination "$backend_plugins"

fixture_lock="$repository_root/Dependencies/fixtures/ci.lock.json"
if [ ! -f "$fixture_lock" ]; then
  echo "Missing fixture lock: $fixture_lock" >&2
  echo "Create and commit the required checksum-pinned CPU CI fixture set before running integration tests." >&2
  exit 1
fi
python3 "$repository_root/Tools/ci/hydrate-fixtures.py" \
  --lock "$fixture_lock" \
  --cache "$KITSUMATE_ARTIFACT_CACHE/fixtures" \
  --repository-root "$repository_root"

echo "Hydrated ONNX Runtime and the required CPU CI fixture set."
echo "The workflow invokes GameCI's unity-test-runner after this step."
