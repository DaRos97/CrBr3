# Code 
solver.py computes the minimum energy configuration for a given set of parameters. The minimization can be either part of a phase
    diagram (PD) or of a magnetization plot (MP), which is given as sys.argv[2]. Thus mainly is used when passing indexes on cluster.
condense_result.py takes all the .npy solutions for a given set of parameters at different gamma and puts them together in the same .hdf5 file.
    It also plots either the PD or the MP, depending again on sys.argv[2].

bao.sh/ygg.sh copies the .hdf5 file or the images (\*.png).
job.sbatch and qjob.qsub are the hpc scripts.

# Folder structure
results is not synched and needs to be in the same folder of solver.py.
results/Phi_values contains the Moire interlayer interaction as well as moire directions and the two lattices.
results/hdf5/ contains all the .hdf5 solutions, one for each moire type.
results/figures/ contains all the figures: phase diagrams (PD) and magnetization plots (MP).
results/phase_diagram_data/ is the main folder of the .npy results.
results/phase_diagram_data/moire_specs(type,eps,ni,gx,gy) depends on the moire structure.
results/phase_diagram_data/moire_specs(type,eps,ni,gx,gy)/gamma/ one directory for each gamma value.

# Computed 
## Bao
    - PD: (rho,d)=(0.1-1.4-5-10-100,0.03-0.0709-0.11-0.2), g=(0 to 2, 50, in Tesla), grid 400, 50 sols (36+18) eps=0.02     (ind 3) -> many missing
## Ygg
    New derivative of order 8
    - DB: (gg,AV)=(200-300-400-500,0-1-2) at (rho,d)=(1.4-100,0.07), g=(from 0 to 1, 50), biaxial 0.03, 100 sols (36+18)        -> results/
    - DB: (gg,AV)=(200-300-400-500,0) at (rho,d)=(1.4-100,0.07), g=(from 0 to 1, 50), biaxial 0.03, 5 sols (36+18)              -> Results/
    Grid 300, new name (no AV), 30 solutions (36+18)
    - PD: (rho,d)=(0.1-1.4-5-10-50-100,0.01-0.03-0.0709-0.11-0.2), g=(0 to 2, 50, in Tesla), eps=0.03
    ------------------------------------------------------------------------------------------------------------------------
    - PD: (rho,d)=(0.1-100,0.01-0.15), g=(0 to 2, 50, in Tesla), eps=0.05-0.005, grid=100
    - PD: (rho,d)=(0.1-100,0.01-0.15), g=(0 to 2, 50, in Tesla), eps=0.05-0.005, grid=200

# Computing
## Bao
## Ygg
    - PD: (rho,d)=(0.1-100,0.01-0.15), g=(0 to 2, 50, in Tesla), eps=0.04-0.03-0.02-0.01, grid=100

# To do

# Comments
If phi_s starts at 0, it stays there for all evolution->we can make it go faster because there is no need to compute dHs.

# Problems
Derivatives (laplacian in particular) is very messy -> blows up the simulation and increasing the averaging does not seem to help.




