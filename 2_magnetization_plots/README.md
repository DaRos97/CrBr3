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
    0 - DFT/exp, uniaxial, all e and n, max_grid = 100, in_pt = 2, gamma from 0 to 3 (100 pts) -> lr -1e-3
    1 - DFT, uniaxial, e=0.1, n=1, max_grid = 200, in_pt = 5, gamma from 0 to 3 (100 pts) -> lr -1e-2 (also t-a). Missing 5pts (13,20,21,33,35)
    2 - DFT, e=0.05, ni=1 and 0.3, max_grid=200, lr = -1e-3.  (MP)
    2 - PD e=0.05, ni=1 and 0.3, max_grid=200, lr = -1e-3.
    2 - PD, e=0.1, ni=1, max_grid=200 and 300, lr = -1e-2, rg=2.
    2 - PD, e=0.1, ni=0.7, max_grid=200 and 300, lr = -1e-2, rg=3.
    - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 200,-1e-1,1    
## Ygg
    0 - DFT/exp, uniaxial, all e and n, max_grid = 200, in_pt = 2, gamma from 0 to 3 (100 pts) -> lr -1e-2
    1 - PD 15x15 for all e-n, gamma = 0. Full=True -> too long to get resources -> only e=0.1 with many missing
    - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 200,-1e-1,2
    - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 300,-1e-1,1
    - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 300,-1e-1,2
    - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 200,-1e-1,3

# Computing
## Bao
    - MP (DFT), e=0.1 & 0.05 & 0.04, ni=1, precision: 200,-1e-1,1
## Ygg
    - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 300,-1e-1,3
    - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 400,-1e-1,1
    - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 400,-1e-1,2
    - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 400,-1e-1,3

# Problems

# Comments
If phi_s starts at 0, it stays there for all evolution->we can make it go faster because there is no need to compute dHs.

# To try
Evidence:
    Solution sometimes seems t-a but just because the phi_s is very rough. In real t-a you can see the domes of phase 0 and pi.
Solution:
    1 - try to end minimization at the minimum dH instead of the converging minimum energy
    2 - higher number of grid points in relation with rg_fit
