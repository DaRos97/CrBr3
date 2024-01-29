import numpy as np
import functions as fs
import sys, os, h5py
from pathlib import Path

"""
Check energy, grad_H and smooth -> derivatives might be wrong
Put grid relative to moire size.
Define moire lattice and gamma wrt cluster input.
"""

machine = fs.get_machine(os.getcwd())

#Maybe define elsewhere the gamma range to consider
input_type,moire_type,moire_pars,gamma = fs.get_parameters(int(sys.argv[1]))
print("Computing with ",input_type," values, moire with ",moire_type," strain of ",moire_pars[moire_type]," and gamma: ",gamma)

#Check if Phi already computed
Phi_fn = fs.get_Phi_fn(moire_type,moire_pars,machine)
if not Path(Phi_fn).is_file():
    print("Computing interlayer coupling...")
    args_Moire = (machine=='loc',moire_type,moire_pars)
    fs.Moire(args_Moire)
with h5py.File(Phi_fn,'r') as f:
    Phi = np.copy(f['Phi'])
    a1_m = np.copy(f['a1_m'])
    a2_m = np.copy(f['a2_m'])

print("Moire lattice vectors: |a_1|=",np.linalg.norm(a1_m),", |a_2|=",np.linalg.norm(a2_m))
print("Constant part of interlayer potential: ",Phi.sum()/Phi.shape[0]/Phi.shape[1]," meV")
#
max_grid = 200 
factor_grid = 100
gridx = int(np.linalg.norm(a1_m)*factor_grid)+1 #make it relative to length of a1_m
gridy = int(np.linalg.norm(a2_m)*factor_grid)+1 #make it relative to length of a2_m
l_g = np.array([gridx,gridy])
if gridx > max_grid or gridy > max_grid:
    i_m = np.argmax(l_g)
    l_g[i_m] = max_grid
    n_m = [np.linalg.norm(a1_m),np.linalg.norm(a2_m)]
    l_g[1-i_m] = max_grid/n_m[i_m]*n_m[1-i_m]
gridx,gridy = l_g
print("Grid size: ",gridx,gridy)

#Compute Phi over new grid parameters
Phi = fs.reshape_Phi(Phi,gridx,gridy)

solution_fn = fs.get_sol_fn(input_type,moire_type,moire_pars,gamma,gridx,gridy,machine)
if not Path(solution_fn).is_file():
    print("Computing magnetization...")
    args_minimization = {
            'args_moire':       (Phi,(a1_m,a2_m)),
            'args_phys':        (fs.rho_phys[input_type],fs.d_phys[input_type]),
            'grid':             (gridx,gridy),
            'learn_rate':       -1e-2,                      #Needs to be negative
            'pts_per_fit':      2,                          #Maybe can be related to gridx/gridy
            'n_initial_pts':    100,                         #64 fixed initial states: n*pi/2 (s and a, n=0..7) + 36 random states
            'maxiter':          1e5, 
            'machine':          machine, 
            'disp':             machine=='loc',
            }
    phi = fs.compute_solution(gamma,args_minimization)
    np.save(solution_fn,phi)
else:
    print("Already computed")













