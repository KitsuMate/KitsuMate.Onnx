#!/usr/bin/env bash
set -euo pipefail

command -v docker >/dev/null
command -v python3 >/dev/null
docker info >/dev/null

if [ "${GITHUB_EVENT_NAME:-}" = "pull_request" ]; then
  echo "Unity jobs must not execute pull-request code." >&2
  exit 1
fi

echo "Unity CI prerequisites are available."
