#!/bin/bash
# Start the optserver simulation service on linux02.
# Usage: ./run_server.sh [circuits/your_circuit.yaml] [port]
cd "$(dirname "$0")"
YAML=${1:-circuits/your_circuit.yaml}
PORT=${2:-8492}
exec ./venv/bin/python -m optserver.app "$YAML" --port "$PORT" --max-jobs "${MAX_JOBS:-4}"
