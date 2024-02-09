import numpy as np
import functions as fs
import sys, os, h5py
from pathlib import Path

machine = fs.get_machine(os.getcwd())

type_computation = 'PD' if machine=='loc' else sys.argv[2]

pd_size = len(fs.rhos)*len(fs.anis)
if type_computation == 'PD':
    moire_type,moire_pars = fs.get_moire_pars(int(sys.argv[1])//pd_size)
    gamma,rho,anisotropy = fs.get_phys_pars(int(sys.argv[1])%pd_size)          
elif type_computation == 'MP':
    input_type,moire_type,moire_pars,gamma = fs.get_MP_pars(int(sys.argv[1]))
    rho = fs.rho_phys[input_type]
    anisotropy = fs.d_phys[input_type]

print("Computing with Moire with ",moire_type," strain of ",moire_pars[moire_type])
print("Physical parameters are gamma: ","{:.4f}".format(gamma),", rho: ","{:.4f}".format(rho),", anisotropy: ","{:.4f}".format(anisotropy))

##############################################################################
max_grid = 200
LR = -1e-2
AV = 2
##############################################################################

#Check if Phi already computed
Phi_fn = fs.get_Phi_fn(moire_type,moire_pars,machine)
if not Path(Phi_fn).is_file():
    print("Computing interlayer coupling...")
    args_Moire = (machine=='loc',moire_type,moire_pars)
    fs.Moire(args_Moire)
Phi = np.load(Phi_fn)
a1_m,a2_m = np.load(fs.get_AM_fn(moire_type,moire_pars,machine))
gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)
precision_pars = (gridx,gridy,LR,AV)

print("Moire lattice vectors: |a_1|=",np.linalg.norm(a1_m),", |a_2|=",np.linalg.norm(a2_m))
print("Constant part of interlayer potential: ",Phi.sum()/Phi.shape[0]/Phi.shape[1]," meV")
print("Grid size: ",gridx,gridy)

#Compute Phi over new grid parameters
Phi = fs.reshape_Phi(Phi,gridx,gridy)

#Check directories for the results exist
fs.check_directory(moire_type,moire_pars,precision_pars,gamma,machine)

solution_fn = fs.get_sol_fn(moire_type,moire_pars,precision_pars,gamma,rho,anisotropy,machine)
if not Path(solution_fn).is_file():
    print("Computing magnetization...")
    args_minimization = {
            'args_moire':       (Phi,(a1_m,a2_m)),
            'args_phys':        (gamma,rho,anisotropy),
            'grid':             (gridx,gridy),
            'learn_rate':       LR,                      #Needs to be negative
            'pts_per_fit':      AV,                          #Maybe can be related to gridx/gridy
            'n_initial_pts':    2,                         #64 fixed initial states: n*pi/2 (s and a, n=0..7) + 36 random states
            'maxiter':          1e5, 
            'machine':          machine, 
            'disp':             machine=='loc',
            }
    phi = fs.compute_solution(args_minimization)
    np.save(solution_fn,phi)
else:
    print("Already computed")













