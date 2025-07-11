#!/bin/bash

# Timeout wrapper for sbatch
# Usage: sbatch_timeout.sh <timeout_seconds> <sbatch_args...>

TIMEOUT=$1
shift

timeout $TIMEOUT sbatch "$@"