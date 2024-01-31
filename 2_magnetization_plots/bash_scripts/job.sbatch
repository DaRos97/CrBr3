#!/bin/bash

#SBATCH --job-name=mp

#SBATCH --partition=shared-cpu
#SBATCH --time=06:00:00

# #SBATCH --partition=debug-cpu
# #SBATCH --time=00:15:00


#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-99


#SBATCH --output=bash_scripts/out.out
#SBATCH --error=bash_scripts/out.err

srun $HOME/2_magnetization_plots/bash_scripts/qjob.qsub $SLURM_ARRAY_TASK_ID $2
