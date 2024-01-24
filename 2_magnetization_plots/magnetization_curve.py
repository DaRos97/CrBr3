import numpy as np
import functions as fs
import sys, os, h5py
from pathlib import Path

input_type = 'DFT'
machine = fs.get_machine(os.getcwd())
#

#Maybe define elsewhere the gamma range to consider
gamma = int(sys.argv[1]) if not machine == 'loc' else 0

#Check if Phi already computed
Phi_fn = fs.get_Phi_fn(machine)
if not Path(Phi_fn).is_file():
    print("Computing interlayer coupling...")
    import Moire
with h5py.File(Phi_fn,'r') as f:
    Phi = np.copy(f['Phi'])
    a1_m = np.copy(f['a1_m'])
    a2_m = np.copy(f['a2_m'])
gridx = 150 #make it relative to length of a1_m
gridy = 100
#Compute Phi over new grid parameters
Phi = fs.reshape_Phi(Phi,gridx,gridy)

solution_fn = fs.get_sol_fn(gamma,machine)
if not Path(solution_fn).is_file():
    print("Computing magnetization...")
    args_minimization = {
            'args_moire':       (Phi,(a1_m,a2_m)),
            'args_phys':        (fs.rho_phys[input_type],fs.d_phys[input_type]),
            'grid':             (gridx,gridy),
            'learn_rate':       -1e-2,                      #Needs to be negative
            'pts_per_fit':      2,                          #Maybe can be related to gridx/gridy
            'n_initial_pts':    64,                         #65 initial states: t-s_pert, 0,pi/2,2pi/2,3pi/2,4pi/2,5pi/2,6pi/2,7pi/2
            'maxiter':          1e1,#5, 
            'machine':          machine, 
            'disp':             machine=='loc',
            }
    phi = fs.compute_solution(gamma,args_minimization)
else:
    phi = np.load(solution_fn)













