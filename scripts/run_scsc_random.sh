#!/usr/bin/env bash
set -e
export PYTHONPATH=src
python -m modal_lookahead.runners.run_scsc_random "$@"
