#!/bin/bash

# Change for multinode config

export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export GPUS_PER_NODE=$SLURM_NTASKS_PER_NODE
if [ -z $GPUS_PER_NODE ]
then
    export GPUS_PER_NODE=1
fi

export MASTER_ADDR=$(scontrol show JobId=$SLURM_JOB_ID | grep BatchHost | tr '=' ' ' | awk '{print $2}')
if [ -z $MASTER_PORT ]
then
    export MASTER_PORT=26845
fi
export RANK=$NODE_RANK
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$(($NODE_RANK%$GPUS_PER_NODE))

export CUDA_VISIBLE_DEVICES=$LOCAL_RANK

#export NCCL_DEBUG=info

#echo $NODE_RANK
#echo $GPUS_PER_NODE
#echo $LOCAL_RANK

exec python -u pretrain.py \
    --balance-strategy swipe \
    --log-interval 100 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --lr-decay-iters 80000 \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --eval-iters 100 \
    --fmoefy
