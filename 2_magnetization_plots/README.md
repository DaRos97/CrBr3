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
    0 - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 200,-1e-1,2
    0 - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 300,-1e-1,1
    0 - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 300,-1e-1,2
    0 - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 200,-1e-1,3
    0 - MP (DFT), e=0.1 & 0.05 & 0.04, ni=1, precision: 200,-1e-1,3
    0 - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 300,-1e-1,3
    0 - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 400,-1e-1,1
    0 - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 400,-1e-1,2
    0 - PDs of e=0.1 & 0.05, ni=1 &0.7 for precision: 400,-1e-1,3
    0 - PD of e=0.04 & 0.03, ni=1 &0.7 for precision: 200,1
    0 - PD of e=0.04 & 0.03, ni=1 &0.7 for precision: 200,2
    0 - PD of e=0.04 & 0.03, ni=1 &0.7 for precision: 300,1
    0 - PD of e=0.04 & 0.03, ni=1 &0.7 for precision: 300,2
    0 - CO (DFT), precision: 200,-1e-1,1
    0 - CO (DFT), precision: 200,-1e-1,2
    #For meeting -> new name (theta)
    1 - MP (0.05,1), (100,1e-1,2) -> no random
    1 - MP (0.05,1), (200,1e-1,0-1-2-3-4-5) -> no random
    1 - MP (0.05,1), (300,1e-1,1-2-3) -> no random
    1 - MP (0.05,1), (400,1e-1,1-2-3) -> no random
    1 - MP (0.05,1), (150,1e-2,1-2) -> defined solutions + 64 fixed solutions (no random) -> not finished low field
    #New name (LR)
    - MP (0.05,1), (100,1-4) -> new method. Only neg in pts (t-s1, t-s2 and t-a)
    - MP (0.05,1), (200,1-4) -> new method. Only neg in pts (t-s1, t-s2 and t-a)

# Computing
## Bao
## Ygg
    - MP (0.05,1), (150,1) -> new method. Neg + 10 constant solutions
    - MP (0.05,1), (150,2) -> new method. Neg + 10 constant solutions
    - MP (0.05,1), (150,3) -> new method. Neg + 10 constant solutions
    - MP (0.05,1), (150,4) -> new method. Neg + 10 constant solutions

    - MP (0.05,1), (160,1) -> new method. Neg + 25 constant solutions
    - MP (0.05,1), (160,2) -> new method. Neg + 25 constant solutions
    - MP (0.05,1), (160,3) -> new method. Neg + 25 constant solutions
    - MP (0.05,1), (160,4) -> new method. Neg + 25 constant solutions

# To do
    - After this set of calculations, see if 300,3 is still good -> if not other tests
    - Compute MP for all eps,ni (20 combinations) -> in DFT values
    - Compute PD for corners (eps=0.05&0.01, ni=1&0.3) -> see if all the same (so no need to compute them all)
    - Take one MP and do many steps close to flip(kink) and flop(jump) as well as for large gamma (to see if it is sharp)

# Comments
If phi_s starts at 0, it stays there for all evolution->we can make it go faster because there is no need to compute dHs.
400 max grid size and 2 pts of average seems good. The learn rate is insignificant now.

# Problems
Some points in the phase diagrams are still oscillating from t-s to t-a
Constant interlayer -> jumps a lot
Solution sometimes seems t-a but just because the phi_s is very rough. In real t-a you can see the domes of phase 0 and pi.





