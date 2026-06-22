#!/bin/bash
# Launch on Slurm cluster
#SBATCH --job-name=max-ai-pretrain
#SBATCH --nodes=3125             # 3125 nodes x 8 GPUs = 25000 H100s
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=90-00:00:00
#SBATCH --partition=gpu

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID

srun --container-image=max-ai:latest \
     --container-mounts=/data:/data,/checkpoints:/checkpoints \
     bash scripts/launch_pretrain.sh
