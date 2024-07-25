#!/bin/bash
echo "num nodes" $SLURM_JOB_NUM_NODES
echo "rank" $SLURM_PROCID 
export ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "Addr" $ADDR

torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=4 --node_rank=$SLURM_PROCID --rdzv_id=32 --rdzv_endpoint=$ADDR:12351 train_model_valid_example.py
