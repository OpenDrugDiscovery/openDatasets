#!/bin/bash -l
#SBATCH --job-name=test_preprocess_sm
#SBATCH --partition=c176
#SBATCH --time=0-12:05:00
#SBATCH --output=slurm/outputs/slurm-%a.out


conda init
conda activate odd

python opendata/preprocessing/peptides.py uniprotkb \
 --min-count 5 --reviewed-only