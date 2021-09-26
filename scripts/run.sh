#!/bin/bash

if [ -z $MOEBENCH_SPACK_LOADED ]
then
    source scripts/spack-env.sh
fi

srun -N 1 --ntasks-per-node 1 --gres=gpu:1 --exclusive --export=ALL \
    ./pretrain_distributed.sh
