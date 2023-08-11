#!/bin/bash -l

#SBATCH --job-name=preprocess_sm
#SBATCH --partition=c112,c176,c60
#SBATCH --array=0-360
#SBATCH --time=0-23:00:00
#SBATCH --output=slurm/outputs/slurm-%A_%a.out


conda activate odd

# each job will see a different ${SLURM_ARRAY_TASK_ID}
echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID} 
# python opendata/preprocessing/small_molecules.py fragmentation -i ${SLURM_ARRAY_TASK_ID} -s 1000000 
# python opendata/preprocessing/small_molecules.py collect-frags
python opendata/preprocessing/small_molecules.py iso-and-tauto -i ${SLURM_ARRAY_TASK_ID} -s 100000 
# python opendata/preprocessing/small_molecules.py collect-iso-tauto-frags