#!/bin/bash
#SBATCH --job-name=oA
#SBATCH --output=oA_%j.out
#SBATCH --error=oA_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2400:00:00

srun --exclusive -N1 -n1 python -u optim_ABZ.py

