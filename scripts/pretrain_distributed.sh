#!/bin/bash

# Change for multinode config

export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export GPUS_PER_NODE=$SLURM_NTASKS_PER_NODE

DATA_PATH=/home/laekov/dataset/enwik8/enwik8_text_document

export MASTER_ADDR=$(scontrol show JobId=$SLURM_JOB_ID | grep BatchHost | tr '=' ' ' | awk '{print $2}')
export MASTER_PORT=26845
export RANK=$NODE_RANK
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$(($NODE_RANK%$GPUS_PER_NODE))

#export NCCL_DEBUG=info

#echo $NODE_RANK
#echo $GPUS_PER_NODE
#echo $LOCAL_RANK

exec python -u pretrain.py \
    --balance-strategy swipe \
    --micro-batch-size 4 \
    --tensor-model-parallel-size 1 \
    --num-layers 12 \
    --hidden-size 2048 \
    --train-iters 3500 \
    --log-interval 1 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --lr-decay-iters 32000 \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --vocab-file /home/cuvelia/aiperf-moebench/data/gpt2-vocab.json \
    --merge-file /home/cuvelia/aiperf-moebench/data/gpt2-merges.txt \
    --split 949,50,1 \
    --distributed-backend nccl \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --eval-interval 500 \
    --eval-iters 100 \
    --fmoefy \
    --num-experts 2 \
    --top-k 2
