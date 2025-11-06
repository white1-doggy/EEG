#!/bin/bash
set -e

CMD=$1
shift || true

if [ "$CMD" == "train" ]; then
  python train.py "$@"
elif [ "$CMD" == "evaluate" ]; then
  python evaluate.py "$@"
else
  echo "Usage: ./run.sh [train|evaluate] <args>"
fi
