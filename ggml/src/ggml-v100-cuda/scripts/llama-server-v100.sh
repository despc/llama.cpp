#!/bin/sh
# Run llama-server with the V100 backend enabled.
# LD_PRELOAD is what points the V100 driver stack at /dev/nvidia-v100-*.
set -e

REDIRECT=/opt/nvidia-v100/lib/v100_redirect.so
BUILD_BIN=$(dirname "$0")/../../../../build-v100/bin

if [ ! -f "$REDIRECT" ]; then
    echo "missing $REDIRECT - see ggml/src/ggml-v100-cuda/README.md" >&2
    exit 1
fi

exec env LD_PRELOAD="$REDIRECT" "$BUILD_BIN/llama-server" "$@"
