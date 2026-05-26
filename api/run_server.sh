#!/usr/bin/env bash
# Start the TackleVision ML inference server.
# Run from the project root: bash api/run_server.sh
set -e
cd "$(dirname "$0")/.."
PYTHON=/Users/pranavr/.pyenv/versions/3.11.6/bin/python3
echo "Starting TackleVision API on http://0.0.0.0:8000 ..."
$PYTHON -m uvicorn api.server:app --host 0.0.0.0 --port 8000
