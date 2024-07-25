# language_model_training

I ran with python 3.9.19, i'll try to give a formula for a conda environment soon

Most important parts:
```
pip install causal-conv1d>=1.4.0
pip install mamba==1.2.2
```

Additionally, follow instructions to install my Rust util for data loading: https://github.com/richardlett/kmer_counter_python_rs 



perlmutter minimal example:

```
salloc --nodes 4 --qos interactive --time 4:00:00 --constraint gpu -A YOUR_ACCOUNT


# load python enviorment


module load nccl

# with this repo as the current directory


time srun --nodes=4 --ntasks-per-node=1 --gpus-per-node=4 jobscript_cross_valid_example.sh
```

For me, it takes around 5 minutes to start actually training when running  on 4 nodes on perlmutter.

if you use more or less GPUS per node, you'll need to adjust the jobscript_cross_valid_example.sh file, along with learning rate or per node batch size.
