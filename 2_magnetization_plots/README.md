# Code 
solver.py computes the minimum energy configuration for a given set of parameters. The minimization can be either part of a phase
    diagram (PD) or of a magnetization plot (MP), which is given as sys.argv[2]. Thus mainly is used when passing indexes on cluster.
condense_result.py takes all the .npy solutions for a given set of moire parameters at different gamma and puts them together in the same .hdf5 file.
the hdf5 structure is: each .hdf5 has a different moire/grid_pts, 

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
    test3
    - PD: ip:0-4-20-24, biaxial im:0-5, grid=100-300-400-500, 50 initial states 30+5
    final
    - PD: ip:0 to 24, biaxial im:0, grid=200, 49 initial states 25+7 from 0 to pi
    - PD: ip:0 to 24, biaxial im:0-1-4, grid=300, 49 initial states 25+7 from 0 to pi
    - PDu: im 0 to 899, grid 400
    Precise:
    - PDb: eps 0.05, rho 1.4, d 0.02
    - PDb: eps 0.01, rho 1.61, d 0.02
    - PDu: tr:0, eps:0.01, rho:1.4, d:0.02
## Ygg
    New name (no AV)
    - PD: (rho,d)=(0.1-1.4-5-10-50-100,0.01-0.03-0.0709-0.11-0.2), g=(0 to 2, 50, in Tesla), eps=0.03, grid 300 (30 sols)
    ------------------------------------------------------------------------------------------------------------------------
    - PD: (rho,d)=(0.1-100,0.01-0.15), g=(0 to 2, 100, in Tesla), eps=0.05 to 0.005, grid=100 -> results_DFT
    - PD: (rho,d)=(0.1-100,0.01-0.15), g=(0 to 2, 50, in Tesla), eps=0.05-0.005, grid=200
    Rescaled results:
    - CO: (rho,d)=(0,0.01-0.03-0.07-0.11-0.15), g=(0 to 0.8, 100), AA and M, grid 100
    - PD: (rho,d)=(0.1-1.4-5-10-100,0.01-0.03-0.07-0.11-0.15), g=(0 to 2, 100, in Tesla), biaxial eps=0.05-0.04-0.03-0.02-0.01-0.005, grid=100
    - PD: (rho,d)=(0.1-1.4-5-10-100,0.01-0.03-0.07-0.11-0.15), g=(0 to 2, 100, in Tesla), uniaxial eps=0.05-0.03-0.02, grid=200 -> many missing
    - PD: (rho,d)=(0.1-1.4-5-10-100,0.01-0.03-0.07-0.11-0.15), g=(0 to 2, 100, in Tesla), uniaxial eps=0.05-0.02, grid=300 -> some missing
    test1
    - PD: (rho,d)=(0.1,0.15), g=(0 to 2, 100, in Tesla), biaxial eps=0.05, grid=100-150-200-250-300-400-500-600-700, 50 initial pts
    - PD: (rho,d)=(0.1,0.15), g=(0 to 2, 100, in Tesla), biaxial eps=0.05, grid=120, 100 initial pts
    - PD: (rho,d)=(0.1,0.15), g=(0 to 2, 100, in Tesla), biaxial eps=0.05, grid=110, 100 random initial pts
    - PD: (rho,d)=(0.1,0.15), g=(0 to 2, 100, in Tesla), biaxial eps=0.05, grid=450-550-600-650-700, 5 initial pts
    test2
    - PD: (rho,d)=(0.1,0.15), g=(0 to 2, 100, in Tesla), biaxial eps=0.05, grid=100-200-300, 10 initial states, high precision (3 solutions at 1e-10 distance)
    - PD: (rho,d)=(0.1,0.15), g=(0 to 2, 100, in Tesla), biaxial eps=0.05, grid=100-200-300, 20 initial states, high precision (2 solutions at 1e-8 distance)
    - PD: (rho,d)=(0.1,0.15), g=(0 to 2, 100, in Tesla), biaxial eps=0.05, grid=110-210-310, 20 initial states, high precision (2 solutions at 1e-8 distance), dH/=2
    - PD: (rho,d)=(0.1-100,0.01-0.15), g=(0 to 2, 100, in Tesla), biaxial eps=0.05-0.005, grid=300, 20 initial states, high precision (2 solutions at 1e-8 distance)    #best -> not good: p4-20 for im5
    - PD: (rho,d)=(0.1-100,0.01-0.15), g=(0 to 2, 100, in Tesla), biaxial eps=0.05-0.005, grid=190, 30 random initial states, precision (3 solutions at 1e-9 distance)
    test3
    - PD: ip:0-4-20-24, uniaxial im:5, grid=300-400, 50 initial states 30+5
    - PD: ip:0-4-20-24, uniaxial im:30, grid=400, 49 initial states 25+7 from 0 to pi
    - PD: ip:4-24, biaxial im:5, grid=100-200, 49 initial states 25+7 from 0 to pi
###################################################################################
###################################################################################
    Final
    - PDb: ip:0 to 24, im:2-3-5, grid=300, 49 initial states 25+7 from 0 to pi

    Precision:
    - PDu: eps 0.01-0.02-0.005-0.003, rho 1.4, d 0.02 -> grid 600, 200 steps

    # grid 400, 200 steps
    - PDb: eps 0.01,        rho 1.4,    d 0.01-0.02
    - PDb: eps 0.05-0.01,   rho 5,      d 0.02

    # grid 900, 200 steps
    - PDu: eps 0.05-0.01-0.001,     rho 1.4,    d 0.02
    - PDu: eps 0.03,                rho 5,      d 0.02
    - PDu: eps 0.05,                rho 5,      d 0.02,     ni=0.1
###################################################################################
###################################################################################
## Maf
    final
    - PDu: im 0 to 899, grid 500

# Computing
## Bao
## Ygg
    Precision:
## Maf

# NB
BIAXIAL:
Main problem of biaxial is for large d at large Moire size. Here the non-smooth behavior of Mz is NOT due to system size (from 300 to 500 same features).
System size though has to be large enough (more than 100, less then 300) for small moire size to be smooth.
Main hypothesis now is that the behavior depends on the initial conditions. The state is stable, there are no big differences between states found in minimization
(they are all twisted-a wrt M region). But for large moire and anisotropy the Mz has large fluctuations between adjacient solutions.
Try small (100) system size with different initial conditions from 0 to pi -> looks better in 4 and 24. Try with larger size (200) -> does not change.
Just keep these jumps.. The jumps of 0.05 (AM=20) disappear going from 200 to 300, so it could be that a dx of 0.067 is needed, which means 3000 pts should
be used for 0.005 (AM=200). Since the features don't change, and the jumps are there just at high anisotropy, try to stick to 300.

Since 0.05 is mostly working, launch full simulations for it. Main problem is that for large rho it takes a lot of time to converge.
Need to fine tune the initial conditions or give a lot of time.
Try small system size with different initial conditions from 0 to pi. If 200 is good just keep that and do also for other im.

UNIAXIAL:
0.05 already very good at 400. Weird behavior at 100,0.15 -> 0 Mz plateau, which could actually be because the high stiffness wants them aligned and the interlayer
is relatively weak. 
Try im 30 in all corners with initial conditions from 0 to pi. -> good ~ 1 min.

Can run all uniaxial (im from 0 to 35) at 500 -> put all gamma of an i_p in same calculation -> 36x25=900 12h calculations to do all of it.
This code is implemented in mafalda and baobab. The job.sh script takes indexes 1-900 because it includes ind_moire(6), ind_tr(6) and i_p(25), while the compressing script
Job.sh takes indexes 1-36 because i_p is not included here.

# Final results remarks
BIAXIAL
Ss expected there are some jumps in Mz for low rho/high d when the moire size increases. Can correct this by going to much larger system size but maybe not worth it.
More notably, some points clearly did not converge for strain 0.05,rho 100, d 0.01-0.03 -> can redo these -> can spot them singularly.

UNIAXIAL
# To do
    - Extract features of biaxial 0.005 strain
    - Extract feautures of uniaxial
    - Plot Mz/Mz^2 (remove initial Mz) wrt field and compare with dG -> find parameters that better reproduce it


