#!/usr/bin/env bash
set -euo pipefail

command -v docker >/dev/null
test -n "${RUNNER_NAME:-}"

if [ "${GITHUB_EVENT_NAME:-}" = "pull_request" ]; then
  echo "Self-hosted Unity jobs must not execute pull-request code." >&2
  exit 1
fi

echo "Worker ${RUNNER_NAME} is available. Artifact hydration is performed by run-unity-tests.sh."
