# language_model_training Minimal example

I ran with python 3.9.19, i'll try to give a formula for a conda environment soon

Most important parts:
```
pip install causal-conv1d>=1.4.0
pip install mamba==1.2.2
```

Additionally, follow instructions to install my Rust util for data loading: https://github.com/richardlett/kmer_counter_python_rs 

You'll also need to download the represesntative and metadata tsvs from GTDB: https://gtdb.ecogenomic.org/downloads.

Note, use the alternative mirror, the primary one is for Austrailia


Extract the tarballs, then edit the following lines in `family_split_validation_loader.py`:

```
GTDB_DOWNLOAD_PATH = "/pscratch/sd/r/richardl/gtdb220_2/gtdb_genomes_reps_r220/database"
BAC_METADATA_PATH = '/pscratch/sd/r/richardl/gtdb220_2/bac120_metadata_r220.tsv'
AR_METADATA_PATH= '/pscratch/sd/r/richardl/gtdb220_2/ar53_metadata_r220.tsv'

```

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

This doesn't use ~108k genomes for a full run, it uses around 75k genomes.

## Notes

AUC is calculated from holdout genomes. See `family_split_validation_loader.py` for sampling details. While we shard training genomes, all nodes have access to all validation genomes (only 500 in total).

The training loop consists of:
1. A training epoch (defined by a fixed number of batches)
2. A validation mega batch

After the validation batch, we calculate the AUC from all pairwise distances as a binary classification problem, using these distances for thresholding. AUC is calculated per rank (GPU), then averaged across all ranks.


Log files outputted (current working directory)

1. `pairwise_distances_histogram.png`:
   Continuously updated plot showing distribution of pairwise distances between genome pairs (same species vs different species).

2. `AUC_log.tsv`:
   TSV file logging last validation AUC and number of gradient steps processed.

3. `log_file.txt`:
   Records training loss and gradient steps. Note: CurriculumFace loss used, so values less interpretable than standard Cross Entropy.

These logs track training progress and the model's ability to distinguish between genomic sequences from same/different species.
