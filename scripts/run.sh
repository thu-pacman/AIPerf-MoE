#!/bin/bash

if [ ! -f config.yaml ]
then
    echo "[Warning] config.yaml is not found, using default config file instead."
    cp config.default.yaml config.yaml
fi

export MASTER_PORT=$(expr $RANDOM % 30000 + 10000)

srun \
    -N 2 \
    --ntasks-per-node=8 \
    --gres=gpu:8 \
    --exclusive \
    --export=ALL \
    scripts/pretrain_distributed.sh
