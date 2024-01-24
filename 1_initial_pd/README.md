# Code structure

All main scripts refer to functions.py and inputs.py.

## Solver -> basic_solver.py

Works with 'basic' interlayer potential of Hejazi et al. as well as with 'general' potential of moirè lattice. 
Inputs.py decides the specs of the minimization. Latest add -> C3 symmetry in unit cell.
When yggdrasil was un-available I wrote the code in 'local_machine_code/' to make easier calculations of selected
points for rough estimates. Is a couple of commits behind the main code.

## Interlayer potential for Moirè lattice -> Moire.py

Works taking the DFT data in 'Data' and computes the interlayer potential by first evaluating the moirè lattice and then looking
at the local stacking configuration of the two layers.
Not sure if it is working with rotations.

## Hysteresis -> hysteresis.py

Should compute the hysteresis cycle starting from various initial conditions, but is not really working.

# Computed stuff in various machines

#Locallly
Args in 'inputs.py' oredr: 20, 1000, 200, 2, -1e-2, basic (AM=20). 
- Index a/b = 208
    Used 7 inputs: t-s_custom, t-a_custom, t-s_pert, const (0 and pi). 
    t-s_pert always solution up to 0.08 which becomes collinear.
    Only transition t-s -> c+ between 0.07 and 0.08.
- Index a/b = 162
    Used 8 inputs: t-s_custom, t-s_pert, t-s2_custom, t-a_custom, const (0 and pi). 
    t-s_pert solution up untill 0.03. From 0.04 to 0.09 it passes to t-s2 got from ansatz 0,pi.
    For 0.42, 0.46, 0.50 stays in t-s2 alligning more and more and becomes collinear at 0.54 and 0.58.
- Index a/b = 248
    Used 5 inputs: t-s_pert, const (0 and pi).
    t-s solution given by t-s_pert untill 0.2, then t-s2 given by 0,pi (small deviation at 0.38, solution of t-s_pert) untill 0.4.
    t-s2 given by both t-s_pert and 0,pi untill c+ at 0.1.    
- Index a/b = 294
    Used 5 inputs: t-s_pert, const (0 and pi).
    t-a solution from 0.20 untill 0.32, then c+ up to 0.4.

# Baobab -> all not symmetrized is in old-res

## Basic Interlayer
###Not Symmetrized
Args in 'inputs.py' oredr: 20, 1000, 200, 2, -1e-2, basic (AM=20)
gamma index: 0,0.01-0.05,0.1-1,2,3,5,7 
Args in 'inputs.py' oredr: 20, 1000, 300, 2, -1e-2, basic (AM=20)
gamma index: 1
Args in 'inputs.py' oredr: 20, 1000, 500, 2, -1e-2, basic (AM=20)
gamma index: 0

index a/b 334: t-a to t-s2 to c+.           max ~ 6
index a/b 81 : t-s to t-s2 to c+.           max ~ 0.2
index a/b 166: t-s to c+.                   max ~ 0.2
index a/b 252: t-s to c+.                   max ~ 0.2
index a/b 246: t-s to t-a to t-s2 to c+.    max ~ 1.7

## General Interlayer
###Not Symmetrized
Args in 'inputs.py' oredr: 20, 1000, 200, 2, -1e-2, general (A2=1.01)
gamma index: 0
Args in 'inputs.py' oredr: 20, 1000, 300, 2, -1e-2, general (A2=1.01)
gamma index: 0
Args in 'inputs.py' oredr: 20, 1000, 500, 2, -1e-2, general (A2=1.01)
gamma index: 0
###Symmetrized
Args in 'inputs.py' oredr: 20, 1000, 200, 2, -1e-2, general (A2=1.01)
gamma index: 0
Args in 'inputs.py' oredr: 20, 1000, 300, 2, -1e-2, general (A2=1.01)
gamma index: 0

# Yggdrasil

## Basic Interlayer
### Symmetrized
index a/b 246: t-s to t-a to t-s2 to c+.    max ~ 1.7
###Not symmetrized
Process 1:
index 210, end 10   
index 231, end 10   
index 252, end 15   
Process 2:
index 103, end 100   
index 106, end 25   
index 207, end 30   
Process 3:
index 246, end 170   
index 263, end 250   
index 266, end 250   
Process 4:
index 290, end 200   
index 312, end 400   
index 350, end 400   
Process 5:
index 317, end 20   
index 337, end 200   
index 358, end 300   

Args in 'inputs.py' oredr: 20, 1000, 100, 2, -1e-2, basic (AM=20)
gamma index: 0
Args in 'inputs.py' oredr: 20, 1000, 200, 2, -1e-2, basic (AM=20)
gamma index: 0
Args in 'inputs.py' oredr: 20, 1000, 300, 2, -1e-2, basic (AM=20)
gamma index: 0
Args in 'inputs.py' oredr: 20, 1000, 400, 2, -1e-2, basic (AM=20)
gamma index: 0
Args in 'inputs.py' oredr: 20, 1000, 500, 2, -1e-2, basic (AM=20)
gamma index: 0

## General Interlayer
Args in 'inputs.py' oredr: 20, 1000, 300, 2, -1e-2, general (A2=1.01)
gamma index: 0
Args in 'inputs.py' oredr: 20, 1000, 500, 2, -1e-2, general (A2=1.01)
gamma index: 0

# Old-results -> Yggdrasil?
Here we used 1/30 steps of gamma. --> pts-gamma = 300

Using 200 points per direction AND a learn rate of 0.01 AND an average done over a distance of 2 sites were computed:
    - 20x20 grid at gamma index = 0,6,12,18,24,30
    - 10x10 grid at gamma index = 1,2,3,4,10,20,25,26,27,29
Using 100 points per direction AND a learn rate of 0.01 AND an average done over a distance of 2 sites were computed:
    - 20x20 grid at gamma index = 0
