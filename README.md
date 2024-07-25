# language_model_training

I ran with python 3.9.19, i'll try to give a formula for a conda environment soon


pip install causal-conv1d>=1.4.0
pip install mamba==1.2.2

perlmutter usage:

```
salloc --nodes 4 --qos interactive --time 4:00:00 --constraint gpu -A YOUR_ACCOUNT


# load python enviorment


module load nccl

# with this repo as the current directory


time srun --nodes=4 --ntasks-per-node=1 --gpus-per-node=4 jobscript_cross_valid_example.py
```

