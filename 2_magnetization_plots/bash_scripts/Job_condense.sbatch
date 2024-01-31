#!/bin/bash

#SBATCH --job-name=c

# #SBATCH --partition=shared-cpu
# #SBATCH --time=00:30:00

#SBATCH --partition=debug-cpu
#SBATCH --time=00:15:00


#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-11

#SBATCH --output=bash_scripts/Out.out
#SBATCH --error=bash_scripts/Out.err

srun $HOME/2_magnetization_plots/bash_scripts/Qjob.qsub $SLURM_ARRAY_TASK_ID $2
