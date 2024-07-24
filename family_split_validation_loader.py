import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from kmer_counter import FastaDataBase

"""
Dataset Loader for Validation Testing

This module creates dataset loaders for validation testing, splitting data based on families.
It ensures validation genomes are in one subset of families, while training genomes are in 
the complementary set.



- Requires at least a couple of nodes for RAM usage without more stringent filtering (Not sure exactly, I always use at least 4)
- Shards training genomes across nodes evenly, maintaining approximate same distribution with regards to training when sampling.
- validation genomes are loaded on every node
- I do some crude phylogenetic balancing to try to get more respresentative results.


Note: Initial plan was to reduce genome count further, but kept as is for comparison purposes. uses around 75 k genomes.
"""



# Path to GTDB reference genomes download and metadata tsvs
# use https://gtdb.ecogenomic.org/downloads
# note: use the mirror, primary is only for Austrailia

GTDB_DOWNLOAD_PATH = "/pscratch/sd/r/richardl/gtdb220_2/gtdb_genomes_reps_r220/database"
BAC_METADATA_PATH = '/pscratch/sd/r/richardl/gtdb220_2/bac120_metadata_r220.tsv'
AR_METADATA_PATH= '/pscratch/sd/r/richardl/gtdb220_2/ar53_metadata_r220.tsv'



def maybe_tqdm(iterable, rank):
    if rank == 0:
        return tqdm(iterable,ncols=80)
    return iterable  # Return the original iterable if not master rank (only want one to print progress bar)



class SplitDNASequencesDataset:
    def __init__(self, n=3000, max_seq_len=2048, spec='genus', filter_csv=None, filter_tax='genus', addit=[], batch_size=128, my_rank=0, num_ranks=1):
        self.n = n
        self.max_seq_len = max_seq_len
        self.spec = spec
        self.filter_csv = filter_csv
        self.filter_tax = filter_tax
        self.addit = addit
        self.batch_size = batch_size
        self.my_rank = my_rank
        self.num_ranks = num_ranks

        self.load_gtdb_data()

        # Sample families
        self.sampled_families, self.total_species = self.sample_families_to_n_species(self.gtdb_tax, self.n)

        # Split genomes into two lists based on families
        self.in_sample_files, self.out_sample_files = self.split_genomes()

        # Create two datasets
        self.in_sample_dataset = self.create_dataset(self.in_sample_files, num_ranks=1,mega_batch=4096)
        self.out_sample_dataset = self.create_dataset(self.out_sample_files,num_ranks=self.num_ranks)

        self.train = DataLoader(self.out_sample_dataset, batch_size=1, shuffle=False)
        self.train_nclasses = len(self.out_sample_files)
    
        self.valid = DataLoader(self.in_sample_dataset, batch_size=1, shuffle=False)
        self.valid_nclasses =  len(self.in_sample_files)

    def load_gtdb_data(self):
        gtdb_df = pd.concat([pd.read_csv(BAC_METADATA_PATH, sep='\t'), 
                             pd.read_csv(AR_METADATA_PATH, sep='\t')]).set_index('accession')
        rep_mask = gtdb_df.index == gtdb_df['gtdb_genome_representative']
        self.gtdb_df = gtdb_df[rep_mask]
        self.gtdb_tax = self.gtdb_df['gtdb_taxonomy'].str.split(';', expand=True)
        self.gtdb_tax.columns = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    def sample_families_to_n_species(self, gtdb_tax, n):
        family_species = gtdb_tax.groupby('family').size()
        families = list(family_species.index)
        # seed needed for consistent across nodes
        random.seed(42)
        random.shuffle(families)
        
        selected_families = []
        total_species = 0
        
        for family in families:
            selected_families.append(family)
            total_species += min(family_species[family],100)
            if total_species > n:
                break
        
        return selected_families, total_species

    def split_genomes(self):
        in_sample_files = []
        out_sample_files = []

        counts = defaultdict(int)
        total_holdout = 0

        for idx, gen in maybe_tqdm(enumerate(self.gtdb_df.index),self.my_rank):
            if self.gtdb_df.iloc[idx]['checkm_contamination'] >= 5:
                continue
            
            file = self.generate_filepath(gen, GTDB_DOWNLOAD_PATH)
            
            if os.path.isfile(file) and file.endswith('.fna.gz'):
                if self.gtdb_tax.iloc[idx]['family'] in self.sampled_families:
                    if counts[self.gtdb_tax.iloc[idx]['genus']] < 10 and total_holdout < 500:
                        in_sample_files.append(file)
                        counts[self.gtdb_tax.iloc[idx]['genus']] = counts[self.gtdb_tax.iloc[idx]['genus']] + 1
                        total_holdout += 1
                else:
                    if counts[self.gtdb_tax.iloc[idx]['genus']] < 20:
                        out_sample_files.append(file)
                        counts[self.gtdb_tax.iloc[idx]['genus']] = counts[self.gtdb_tax.iloc[idx]['genus']] + 1

        return in_sample_files, out_sample_files
    def generate_filepath(self, accession, prefix):
        accession_clean = accession[3:]
        
        parts = accession_clean.split('_')
        prefix_code = parts[0]
        number = parts[1]

        subdir1 = number[:3]
        subdir2 = number[3:6]
        subdir3 = number[6:9]

        filepath = f"{prefix}/{prefix_code}/{subdir1}/{subdir2}/{subdir3}/{accession_clean}_genomic.fna.gz"
        
        return filepath
    def create_dataset(self, file_list, num_ranks,mega_batch=1048576 // 256):
        return DNASequencesDataset(file_list, len(file_list), self.max_seq_len, self.spec, 
                                   self.filter_csv, self.filter_tax, self.addit, 
                                   self.batch_size, self.my_rank if num_ranks != 1 else 0, num_ranks,mega_batch=1048576 // 256)
class DNASequencesDataset(Dataset):
    def __init__(self, file_list, n_classes, max_seq_len=2048, spec=None, filter_csv=None, filter_tax='genus', addit=[], batch_size=128, my_rank=0, num_ranks=1, mega_batch=1048576 // 256):
        self.file_list = file_list
        self.n_classes = n_classes
        self.max_seq_len = max_seq_len
        self.mega_batch_size = mega_batch
        if my_rank == 0:
            print("Found", len(file_list))
            print(f"Loading {len(file_list)} files, {len(file_list)/num_ranks} per rank ({num_ranks} ranks), this may take a hot minute")


        self.db = FastaDataBase(file_list, 2001, my_rank, num_ranks)
        self.n_classes = len(file_list)

    def __len__(self):
        return self.mega_batch_size

    def __getitem__(self, idx):
        aaq = self.db.sample_beta(1048576 // 256, 2000)  # Adjust as necessary

        m = torch.nn.functional.one_hot(torch.tensor(aaq[-1].astype(np.int64)), num_classes=self.n_classes ).float()
        
        adjusted_max_seq_len = 2000

        # this is dead code mostly
        # Pad sequences to the fixed length and create attention masks
        padded_texts = np.zeros((1048576 // 256, 2000), dtype=int)
        attention_masks = np.ones((1048576 // 256, 500), dtype=int)
        attention_masks[:self.mega_batch_size, :2000] = 1

        texts = np.reshape(aaq[0], (-1, 2000)) #+ 65
        padded_texts[:self.mega_batch_size,:2000] = texts
        padded_texts = torch.tensor(padded_texts, dtype=torch.int32)
        attention_masks = torch.tensor(attention_masks, dtype=torch.int32)

        return (m, padded_texts, attention_masks), m



def create_data_loader(file_list, n_classes=10000000, batch_size=128, max_seq_len=2048,my_rank=0, num_ranks=1):
    dataset = DNASequencesDataset(file_list, n_classes, max_seq_len,my_rank=my_rank, num_ranks=num_ranks,)#addit=["/pscratch/sd/r/richardl/RefSeq-Representatives/protists/","/pscratch/sd/r/richardl/RefSeq-Representatives/fungi/"])
    return DataLoader(dataset, batch_size=1, shuffle=False), dataset.n_classes
