I've copyedited your README markdown file. Here's the revised version:

# language_model_training Minimal Example

## Setup 

This project was developed using Python 3.9.19. A conda environment formula will be provided soon.

Most important dependencies:
```
pip install causal-conv1d>=1.4.0
pip install mamba==1.2.2
```

Additionally:
1. Install the Rust utility for data loading: https://github.com/richardlett/kmer_counter_python_rs
2. Download the representative and metadata TSVs from GTDB: https://gtdb.ecogenomic.org/downloads
   Note: Use the alternative mirror; the primary one is for Australia.
3. Extract the tarballs, then edit the following lines in `family_split_validation_loader.py`:
   ```python
   GTDB_DOWNLOAD_PATH = "/pscratch/sd/r/richardl/gtdb220_2/gtdb_genomes_reps_r220/database"
   BAC_METADATA_PATH = '/pscratch/sd/r/richardl/gtdb220_2/bac120_metadata_r220.tsv'
   AR_METADATA_PATH = '/pscratch/sd/r/richardl/gtdb220_2/ar53_metadata_r220.tsv'
   ```
## Running
Perlmutter minimal example:
```bash
salloc --nodes 4 --qos interactive --time 4:00:00 --constraint gpu -A YOUR_ACCOUNT
# Load Python environment
module load nccl
# With this repo as the current directory
time srun --nodes=4 --ntasks-per-node=1 --gpus-per-node=4 jobscript_cross_valid_example.sh
```

It takes approximately 5 minutes to start training when running on 4 nodes on Perlmutter.
If you use more or fewer GPUs per node, adjust the `jobscript_cross_valid_example.sh` file, along with the learning rate or per-node batch size.
This example uses around 75k genomes instead of the full ~108k genomes.

## Notes

AUC is calculated from holdout genomes. See `family_split_validation_loader.py` for sampling details. While we shard training genomes, all nodes have access to all validation genomes (only 500 in total).

The training loop consists of:
1. A training epoch (defined by a fixed number of batches)
2. A validation mega batch

After the validation batch, we calculate the AUC from all pairwise distances as a binary classification problem (same vs different genome), using these distances for thresholding. AUC is calculated per rank (GPU), then averaged across all ranks.

Log files outputted (current working directory):
1. `pairwise_distances_histogram.png`:
   Continuously updated plot showing distribution of pairwise distances between genome pairs (sequences from same species vs different species ).
2. `AUC_log.tsv`:
   TSV file logging last validation AUC and number of gradient steps processed.
3. `log_file.txt`:
   Records training loss and gradient steps. Note: CurriculumFace loss is used, so values are less interpretable than standard Cross Entropy.
