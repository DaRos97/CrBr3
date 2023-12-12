# Title

# To do
Update git -> changed cluster info.
Look at 200_general 0 -> look at single solutions and if not good try singularly locally with different parameters.
Write script to compute many gamma of particular points in PD -> first for basic.
Try new things on hysteresis.

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

# Baobab

## Basic Interlayer
Args in 'inputs.py' oredr: 20, 1000, 200, 2, -1e-2, basic (AM=20)
gamma index: 0,0.01-0.09(..) ,0.1-1,2,3,5,7 
Args in 'inputs.py' oredr: 20, 1000, 300, 2, -1e-2, basic (AM=20)
gamma index: 1
Args in 'inputs.py' oredr: 20, 1000, 500, 2, -1e-2, basic (AM=20)
gamma index: 0

index a/b 334: gamma 0 to 10. t-a to t-s2 to c+.
index a/b 81 : 
index a/b 166: 
index a/b 252: 
index a/b 246: 

## General Interlayer
Args in 'inputs.py' oredr: 20, 1000, 200, 2, -1e-2, general (A2=1.01)
gamma index: 0
Args in 'inputs.py' oredr: 20, 1000, 300, 2, -1e-2, general (A2=1.01)
gamma index: 0
Args in 'inputs.py' oredr: 20, 1000, 500, 2, -1e-2, general (A2=1.01)
gamma index: 0

# Yggdrasil

## Basic Interlayer
Args in 'inputs.py' oredr: 20, 1000, 100, 2, -1e-2, basic (AM=20)
gamma index: 0
Args in 'inputs.py' oredr: 20, 1000, 200, 2, -1e-2, basic (AM=20)
gamma index: 0(..)
Args in 'inputs.py' oredr: 20, 1000, 300, 2, -1e-2, basic (AM=20)
gamma index: 0(..)
Args in 'inputs.py' oredr: 20, 1000, 400, 2, -1e-2, basic (AM=20)
gamma index: 0(..)
Args in 'inputs.py' oredr: 20, 1000, 500, 2, -1e-2, basic (AM=20)
gamma index: 0(..)

## General Interlayer
Args in 'inputs.py' oredr: 20, 1000, 300, 2, -1e-2, general (A2=1.01)
gamma index: 0(..)
Args in 'inputs.py' oredr: 20, 1000, 500, 2, -1e-2, general (A2=1.01)
gamma index: 0(..)

# Old-results
Here we used 1/30 steps of gamma. --> pts-gamma = 300

Using 200 points per direction AND a learn rate of 0.01 AND an average done over a distance of 2 sites were computed:
    - 20x20 grid at gamma index = 0,6,12,18,24,30
    - 10x10 grid at gamma index = 1,2,3,4,10,20,25,26,27,29
Using 100 points per direction AND a learn rate of 0.01 AND an average done over a distance of 2 sites were computed:
    - 20x20 grid at gamma index = 0
