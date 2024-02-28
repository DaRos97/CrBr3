import numpy as np
import functions as fs
import sys, os, h5py
from pathlib import Path

ind = int(sys.argv[1])
##############################################################################
max_grid = 150
AV = 1
##############################################################################
S = 3/2
machine = fs.get_machine(os.getcwd())

type_computation = 'PD' if len(sys.argv)<3 else sys.argv[2]

if type_computation == 'PD':
    moire_type,moire_pars = fs.get_moire_pars(ind)
    gamma,rho,anisotropy = fs.get_phys_pars(ind,'MPs')          
elif type_computation == 'MP':
    input_type,moire_type,moire_pars,gamma = fs.get_MP_pars(ind,'MPs')
    rho = fs.rho_phys[input_type]
    anisotropy = fs.d_phys[input_type]
    print("Input type: ",input_type)
elif type_computation == 'CO':
    rho = 1000
    ind_a = ind // (2*len(fs.gammas['M']))
    ind_l = ind % (2*len(fs.gammas['M']))
    anisotropy = fs.anis[ind_a]
    #Two cases: AA and M
    list_interlayer = ['AA','M']
    place_interlayer = list_interlayer[ind_l//len(fs.gammas['M'])]
    gamma = fs.gammas[place_interlayer][ind_l%len(fs.gammas['M'])]
    moire_type = 'const'
    moire_pars = {}
    moire_pars[moire_type] = {'place':place_interlayer,}
    moire_pars['theta'] = 0.
elif type_computation == 'DB':
    ggg = [100,200,300,400,500]
    avav = [0,1,2,3,4]
    max_grid = ggg[ind // (5*100)]
    AV = avav[ind % (5*100) //100]
    rho = 100#1.4
    anisotropy = 0.0709
    gamma = fs.gammas['MPs'][ind % (5*100) %100]
    moire_type,moire_pars = fs.get_moire_pars(0)

print("Computing with Moire with ",moire_type," strain of ",moire_pars[moire_type]," and rotation ",moire_pars['theta'])
print("Physical parameters are gamma: ","{:.4f}".format(gamma),", rho: ","{:.4f}".format(rho),", anisotropy: ","{:.4f}".format(anisotropy))

#Check if Phi already computed
Phi_fn = fs.get_Phi_fn(moire_type,moire_pars,machine)
if not Path(Phi_fn).is_file():
    print("Computing interlayer coupling...")
    args_Moire = (machine=='loc',moire_type,moire_pars)
    fs.Moire(args_Moire)
#Try a couple of times to load Phi since sometimes it does not work
Phi,a1_m,a2_m = fs.load_Moire(Phi_fn,fs.get_AM_fn(moire_type,moire_pars,machine))
#######
Phi /= 2*(S**2)
####### 

gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)
precision_pars = (gridx,gridy,AV)

print("Moire lattice vectors: |a_1|=",np.linalg.norm(a1_m),", |a_2|=",np.linalg.norm(a2_m))
print("Relative angle (deg): ",180/np.pi*np.arccos(np.dot(a1_m/np.linalg.norm(a1_m),a2_m/np.linalg.norm(a2_m))))
print("Constant part of interlayer potential: ",Phi.sum()/Phi.shape[0]/Phi.shape[1]," meV")
print("Grid size: ",gridx,'x',gridy,', average: ',AV)

if 0 and machine =='loc':
    exit()
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
            'pts_per_fit':      AV,                          #Maybe can be related to gridx/gridy
            'n_initial_pts':    100,                         #three solution initial states, 25 constant initial states and n-25 random states
            'maxiter':          1e5, 
            'machine':          machine, 
            'disp':             machine=='loc',
            'type_comp':        type_computation,
            }
    phi = fs.compute_solution(args_minimization)
    if not machine == 'loc':
        np.save(solution_fn,phi)
else:
    print("Already computed")













