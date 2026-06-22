#!/bin/bash
set -e
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=8 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=29500 \
  training/pretrain.py
