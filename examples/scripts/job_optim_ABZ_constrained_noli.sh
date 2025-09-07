#!/bin/bash
#SBATCH --job-name=oA_c_nl
#SBATCH --output=oA_c_nl_%j.out
#SBATCH --error=oA_c_nl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2400:00:00

srun --exclusive -N1 -n1 python -u optim_ABZ_constrained_noli.py

