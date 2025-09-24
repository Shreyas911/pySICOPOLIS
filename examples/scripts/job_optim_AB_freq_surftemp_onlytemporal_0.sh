#!/bin/bash
#SBATCH --job-name=oAB_f_st_ot_0
#SBATCH --output=oAB_f_st_ot_0_%j.out
#SBATCH --error=oAB_f_st_ot_0_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2400:00:00

srun --exclusive -N1 -n1 python -u optim_AB_freq_surftemp_onlytemporal_0.py

