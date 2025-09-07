#!/bin/bash
#SBATCH --job-name=oA_r
#SBATCH --output=oA_r_%j.out
#SBATCH --error=oA_r_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2400:00:00

srun --exclusive -N1 -n1 python -u optim_ABZ_reduced.py

