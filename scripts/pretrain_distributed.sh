#!/bin/bash

# Change for multinode config

export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export GPUS_PER_NODE=$SLURM_NTASKS_PER_NODE

DATA_PATH=/home/cuvelia/aiperf-moebench/data/enwik8/enwik8_text_document

export MASTER_ADDR=$(scontrol show JobId=$SLURM_JOB_ID | grep BatchHost | tr '=' ' ' | awk '{print $2}')
export MASTER_PORT=26845
export RANK=$NODE_RANK
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$(($NODE_RANK%$GPUS_PER_NODE))

#export NCCL_DEBUG=info

#echo $NODE_RANK
#echo $GPUS_PER_NODE
#echo $LOCAL_RANK

exec python pretrain.py \
       --num-layers 4 \
       --hidden-size 512 \
       --num-attention-heads 16 \
       --micro-batch-size 2 \
       --global-batch-size 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 1000 \
       --lr-decay-iters 320000 \
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
       --checkpoint-activations \
       --log-interval 1 \
       --save-interval 10000000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fmoefy \
       --num-experts 4 \
       --top-k 2 \
