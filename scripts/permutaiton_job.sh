#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:0

module load anaconda3
source activate torch


cd ../
python permutation.py

source deactivate