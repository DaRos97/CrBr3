#!/bin/bash

#SBATCH --job-name=mp

# #SBATCH --partition=shared-cpu
# #SBATCH --time=00:30:00

#SBATCH --partition=debug-cpu
#SBATCH --time=00:10:00


#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-11

#SBATCH --output=out.out
#SBATCH --error=out.err

srun qjob.qsub $SLURM_ARRAY_TASK_ID
