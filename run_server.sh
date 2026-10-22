#!/usr/bin/env bash
# Starts the analysis service on Linux/macOS.
#
#   ./run_server.sh                 listens on 127.0.0.1:4445 (use with a Cloudflare tunnel)
#   PORT=5000 ./run_server.sh       another port
#   HOST=0.0.0.0 ./run_server.sh    also accept connections from the network
#   ANALYSE_API_KEY=secret ./run_server.sh
#
# Needs the virtual environment first - see DEPLOY.md.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "No virtual environment found. Create it first - see DEPLOY.md." >&2
  exit 1
fi

exec .venv/bin/python server.py
