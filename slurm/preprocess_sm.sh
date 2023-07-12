#!/bin/bash -l

#####################
# job-array example #
#####################

#SBATCH --job-name=test_preprocess_sm
#SBATCH --partition=c112,c176
#SBATCH --array=2-240
#SBATCH --time=0-24:05:00
#SBATCH --output=slurm/outputs/slurm-%A_%a.out


conda init bash
conda activate odd

# each job will see a different ${SLURM_ARRAY_TASK_ID}
echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID} 
# python opendata/preprocessing/small_molecules.py fragmentation --chunk-id ${SLURM_ARRAY_TASK_ID} --chunk-size 1000000 
# python opendata/preprocessing/small_molecules.py collect-frags
python opendata/preprocessing/small_molecules.py iso-and-tauto --chunk-id ${SLURM_ARRAY_TASK_ID} --chunk-size 500000 
