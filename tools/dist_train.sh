#!/usr/bin/env bash
set -eux

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
PYTHONPATH="$(dirname $0)/.."  \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
