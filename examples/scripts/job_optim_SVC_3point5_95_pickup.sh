#!/bin/bash
#SBATCH --job-name=oS3p5_95_p
#SBATCH --output=oS3p5_95_p_%j.out
#SBATCH --error=oS3p5_95_p_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2400:00:00

srun --exclusive -N1 -n1 python -u optim_SVC_3point5_95_pickup.py

