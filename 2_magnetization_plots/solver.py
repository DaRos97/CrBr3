import numpy as np
import functions as fs
import sys, os, h5py
from pathlib import Path

"""
Check energy, grad_H and smooth -> derivatives might be wrong
Put grid relative to moire size.
Define moire lattice and gamma wrt cluster input.
"""

Full = False
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
Phi = np.load(Phi_fn)
a1_m,a2_m = np.load(fs.get_AM_fn(moire_type,moire_pars,machine))
max_grid = 200 if Full else 100
gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)

print("Moire lattice vectors: |a_1|=",np.linalg.norm(a1_m),", |a_2|=",np.linalg.norm(a2_m))
print("Constant part of interlayer potential: ",Phi.sum()/Phi.shape[0]/Phi.shape[1]," meV")
print("Grid size: ",gridx,gridy)

#Compute Phi over new grid parameters
Phi = fs.reshape_Phi(Phi,gridx,gridy)

#Check directories for the results exist
fs.check_directory(input_type,moire_type,moire_pars,gamma,gridx,gridy,machine)

solution_fn = fs.get_sol_fn(input_type,moire_type,moire_pars,gamma,gridx,gridy,machine)
if not Path(solution_fn).is_file():
    print("Computing magnetization...")
    args_minimization = {
            'args_moire':       (Phi,(a1_m,a2_m)),
            'args_phys':        (fs.rho_phys[input_type],fs.d_phys[input_type]),
            'grid':             (gridx,gridy),
            'learn_rate':       -1e-2,                      #Needs to be negative
            'pts_per_fit':      2,                          #Maybe can be related to gridx/gridy
            'n_initial_pts':    64 if Full else 32,                         #64 fixed initial states: n*pi/2 (s and a, n=0..7) + 36 random states
            'maxiter':          1e5, 
            'machine':          machine, 
            'disp':             machine=='loc',
            }
    phi = fs.compute_solution(gamma,args_minimization)
    if not machine == 'loc':
        np.save(solution_fn,phi)
else:
    print("Already computed")













