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
    #new name
    3 - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 200,-1e-1,1    
    3 - MP (DFT), e=0.1 & 0.05 & 0.04, ni=1, precision: 200,-1e-1,3
    #new name (LR)
## Ygg
    0 - CO AA&M d=0.07-0.1-0.2    150,1   25 const solutions, rho=0
    0 - PD (0.03,1), (rho,d):(1.1%2,0%0.2), 13x13
    0 - MP (0.05-0.03-0.01-0.005,1-0.5-0.3), g:(0%3,100) (d,rho) = (0.0709,1.4) (no 0.005,0.3)
    0 - MP (0.05-0.03-0.01-0.005,1-0.5-0.3), g:(0%3,100) (d,rho) = (0.1,1.4) (no 0.005,0.3)
    0 - MP (0.05-0.03-0.01-0.005,1-0.5-0.3), g:(0%3,100) (d,rho) = (0.2,1.4) (no 0.005,0.3)

# Computing
## Bao
## Ygg
All in (150,1,25 sol)
    - PD rho:(0.1,10,25), d:(0.01,0.07,0.1,0.2), g:(0,2,100)    --> error in d: twice the value --> delet it

# To do
    - PD rho:(0.1,10,25), d:(0.01,0.07,0.1,0.2), g:(0,2,100) ('12', correct d)
    - Try some MP with rho different for 1 and 2

# Comments
If phi_s starts at 0, it stays there for all evolution->we can make it go faster because there is no need to compute dHs.
Stay for meeting of 19/02 at grid 150 and AV=1.

# Problems
Some points in the phase diagrams are still oscillating from t-s to t-a
Constant interlayer -> jumps a lot
Solution sometimes seems t-a but just because the phi_s is very rough. In real t-a you can see the domes of phase 0 and pi.





