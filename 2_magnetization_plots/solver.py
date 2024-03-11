import numpy as np
import functions as fs
import sys, os, h5py
from pathlib import Path

ind = int(sys.argv[1])
##############################################################################
machine = fs.get_machine(os.getcwd())
rescaled = True

type_computation = 'PD' if len(sys.argv)<3 else sys.argv[2]
ind_moire = 0 if len(sys.argv)<4 else int(sys.argv[3])

if type_computation[:2] == 'PD':
    if type_computation == 'PDb':            #Phase Diagram biaxial
        max_grid = 100
        moire_pars = {
            'type':'biaxial',
            'eps':fs.epss[ind_moire],       
            'theta':fs.thetas,
            }
    if type_computation == 'PDu':            #Phase Diagram uniaxial
        max_grid = 300
        moire_pars = {
            'type':'uniaxial',
            'eps':fs.epss[ind_moire],
            'ni':0,
            'phi':0,
            'theta':fs.thetas,
            }
    l_a = len(fs.anis)
    l_g = len(fs.gammas['MPs'])
    rho = fs.rhos[ind // (l_a*l_g)]
    anisotropy = fs.anis[ind % (l_a*l_g) // l_g]
    gamma = fs.gammas['MPs'][ind % (l_a*l_g) % l_g]
elif type_computation == 'CO':          #Constant interlayer interaction
    max_grid = 100
    rho = 0
    ind_a = ind // (2*len(fs.gammas['M']))
    ind_l = ind % (2*len(fs.gammas['M']))
    anisotropy = fs.anis[ind_a]
    #Two cases: AA and M
    list_interlayer = ['AA','M']
    place_interlayer = list_interlayer[ind_l//len(fs.gammas['M'])]
    gamma = fs.gammas[place_interlayer][ind_l%len(fs.gammas['M'])]
    moire_pars = {
        'type':'const',
        'place':place_interlayer,
        'theta':fs.thetas,
        }
elif type_computation == 'DB':          #Debug mode, to see which values are better for the minimization
    ggg = [100,200,300,400,500]
    g_pts = len(fs.gammas['MPs'])
    max_grid = ggg[ind // g_pts]
    rho = 1.4
    anisotropy = 0.0709
    gamma = fs.gammas['MPs'][ind % g_pts]
    moire_pars = {
        'type':'biaxial',
        'eps':fs.epss[ind_moire],       
        'theta':fs.thetas,
        }

phys_args = (gamma,rho,anisotropy)

print("Computing Moire: ",moire_pars)
print("Physical parameters are gamma: ","{:.4f}".format(gamma),", rho: ","{:.4f}".format(rho),", anisotropy: ","{:.4f}".format(anisotropy))

#Check if Phi already computed
Phi_fn = fs.get_Phi_fn(moire_pars,machine,rescaled)
if not Path(Phi_fn).is_file():
    print("Computing interlayer coupling...")
    fs.Moire(moire_pars,machine,rescaled)
#Try a couple of times to load Phi since sometimes it does not work
Phi,a1_m,a2_m = fs.load_Moire(Phi_fn,moire_pars,machine)
#######

gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)
grid_pts = (gridx,gridy)

print("Moire lattice vectors: |a_1|=",np.linalg.norm(a1_m),", |a_2|=",np.linalg.norm(a2_m))
print("Relative angle (deg): ","{:.2f}".format(180/np.pi*np.arccos(np.dot(a1_m/np.linalg.norm(a1_m),a2_m/np.linalg.norm(a2_m)))))
print("Grid size: ",gridx,'x',gridy)
if moire_pars['type'] == 'const':
    print("Constant interlayer: ","{:.5f}".format(Phi.sum()/Phi.shape[0]/Phi.shape[1]))

if 0 and machine =='loc':
    exit()
#Compute Phi over new grid parameters
Phi = fs.reshape_Phi(Phi,gridx,gridy)

#Check directories for the results exist
fs.check_directory(moire_pars,grid_pts,gamma,machine)

solution_fn = fs.get_sol_fn(moire_pars,grid_pts,phys_args,machine)
if not Path(solution_fn).is_file():
    print("Computing magnetization...")
    args_minimization = {
            'args_moire':       (Phi,(a1_m,a2_m)),
            'args_phys':        phys_args,
            'grid':             grid_pts,
            'n_initial_pts':    len(fs.list_ind[type_computation]),                         #three solution initial states, 25 constant initial states and n-25 random states
            'maxiter':          1e5, 
            'machine':          machine, 
            'disp':             machine=='loc',
            'type_comp':        type_computation,
            }
    phi = fs.compute_solution(args_minimization)
    if 1:#not machine == 'loc':
        np.save(solution_fn,phi)
else:
    print("Already computed")













