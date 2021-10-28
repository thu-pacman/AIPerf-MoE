#!/bin/bash

if [ -z $MOEBENCH_SPACK_LOADED ]
then
    source scripts/spack-env.sh
fi

if [ ! -f config.yaml ]
then
    echo "[Warning] config.yaml is not found, using default config file instead."
    cp config.default.yaml config.yaml
fi

srun -p Big -A priority -N 1 --ntasks-per-node 8 --gres=gpu:8 --exclusive --export=ALL \
    scripts/pretrain_distributed.sh
