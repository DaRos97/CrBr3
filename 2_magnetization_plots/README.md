# Code 
solver.py computes the min energy configuration for a given set of parameters.
condense_result.py takes all the .npy solutions for a given set of parameters at different gamma and puts them together in the same .hdf5 file.
plot_result.py computes the magnetization plot.

bao.sh/ygg.sh copies the .hdf5 file.
job.sbatch/qjob.qsub are the hpc scripts.

# Folder structure
Phi_values is not synched and needs to be in the same folder of solver.py
results is not synched and needs to be in the same folder of solver.py
Locally, results/hdf5/ contains all the .hdf5 solutions.
Remotely (cluster), results/ contains folders with the same name of the moire parameters, which in turn contain both the .npy gamma files and the .hdf5 condensed file.

# Computed 

# Computing
## Bao
    - uniaxial, e=0.1, n=1, max_grid = 200, in_pt = 100, gamma from 0 to 3 (100 pts)
    - uniaxial, e=0.05, n=1, max_grid = 100, in_pt = 32, gamma from 0 to 3 (100 pts)
