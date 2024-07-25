#!/bin/bash
echo "num nodes" $SLURM_JOB_NUM_NODES
echo "rank" $SLURM_PROCID 
export ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "Addr" $ADDR

#source ~/.bashrc_backup
#mamba activate /pscratch/sd/r/richardl/gnn_binning_cad2
#module load nccl
#export ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=4 --node_rank=$SLURM_PROCID --rdzv_id=32 --rdzv_endpoint=$ADDR:12351 cross_valid_example.py
