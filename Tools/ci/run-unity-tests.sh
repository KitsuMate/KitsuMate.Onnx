#!/usr/bin/env bash
set -euo pipefail

: "${KITSUMATE_ARTIFACT_CACHE:?Set a writable cache directory for checksum-pinned artifacts.}"
: "${KITSUMATE_FIXTURE_PROFILE:?Select a checksum-pinned external fixture profile.}"
: "${KITSUMATE_UNITY_COMMAND:?Set the reviewed GameCI Unity test command on the protected runner.}"

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
backend_plugins="$repository_root/Packages/ai.kitsumate.onnx.backend.onnxruntime/Runtime/Plugins"

python3 "$repository_root/Tools/ci/hydrate-onnxruntime.py" \
  --lock "$repository_root/Dependencies/onnxruntime.lock.json" \
  --cache "$KITSUMATE_ARTIFACT_CACHE" \
  --destination "$backend_plugins"

fixture_lock="$repository_root/Dependencies/fixtures/${KITSUMATE_FIXTURE_PROFILE}.lock.json"
if [ ! -f "$fixture_lock" ]; then
  echo "Missing fixture lock: $fixture_lock" >&2
  echo "Create and commit the explicit checksum-pinned profile before running integration tests." >&2
  exit 1
fi
python3 "$repository_root/Tools/ci/hydrate-fixtures.py" \
  --lock "$fixture_lock" \
  --cache "$KITSUMATE_ARTIFACT_CACHE/fixtures" \
  --repository-root "$repository_root"

mkdir -p "$repository_root/TestResults"
cd "$repository_root"
eval "$KITSUMATE_UNITY_COMMAND"
