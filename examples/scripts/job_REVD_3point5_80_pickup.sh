#!/bin/bash
#SBATCH --job-name=R3p5_80_p
#SBATCH --output=R3p5_80_p_%j.out
#SBATCH --error=R3p5_80_p_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2400:00:00

srun --exclusive -N1 -n1 python -u REVD_3point5_80_pickup.py

