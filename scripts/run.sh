#!/bin/bash

if [ -z $MOEBENCH_SPACK_LOADED ]
then
    source scripts/spack-env.sh
fi

srun -N 1 --ntasks-per-node 8 --gres=gpu:8 --exclusive --export=ALL \
    scripts/pretrain_distributed.sh
