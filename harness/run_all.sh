#!/usr/bin/env bash
# Convenience runner. Edit MAS_LLM_MODE to switch backends.
set -e
export MAS_LLM_MODE="${MAS_LLM_MODE:-mock}"
echo "Backend: $MAS_LLM_MODE"
python3 run_experiment.py --seeds "${1:-30}" --trials "${2:-20}"
