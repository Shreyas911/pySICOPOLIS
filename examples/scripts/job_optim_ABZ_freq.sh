#!/bin/bash
#SBATCH --job-name=oA_f
#SBATCH --output=oA_f_%j.out
#SBATCH --error=oA_f_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2400:00:00

srun --exclusive -N1 -n1 python -u optim_ABZ_freq.py

