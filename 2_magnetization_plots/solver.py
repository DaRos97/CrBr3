import numpy as np
import functions as fs
import sys, os, h5py
from pathlib import Path
import getopt
from time import time
t0 = time()

##############################################################################
machine = fs.get_machine(os.getcwd())
rescaled = True
##############################################################################
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "i:",["type=","i_m=","max_grid=","ni="])
    ind = 0
    type_computation = 'PDb'
    ind_moire = 0
    max_grid = 300
    ind_ni = 0
except:
    print("Error in inputs")
    exit()
for opt, arg in opts:
    if opt in ['-i']:
        ind = int(arg)
    if opt == '--type':
        type_computation = arg
    if opt == '--i_m':
        ind_moire = int(arg)
    if opt == '--max_grid':
        max_grid = int(arg)
    if opt == '--ni':
        ind_ni = float(arg)
if type_computation[:2] == 'PD':
    if type_computation == 'PDb':            #Phase Diagram biaxial
        moire_pars = {
            'type':'biaxial',
            'eps':fs.epss[ind_moire],       
            'theta':fs.thetas,
            }
    if type_computation == 'PDu':            #Phase Diagram uniaxial
        ind_m = ind_moire//len(fs.translations)
        ind_tr = ind_moire%len(fs.translations)
        moire_pars = {
            'type':'uniaxial',
            'eps':0.04,#fs.epss[ind_m],
            'ni':0.1,#ind_ni,#fs.nis[ind_ni],
            'phi':0*np.pi,
            'tr':0,#fs.translations[ind_tr],
            'theta':fs.thetas,
            }
    #Extract gamma,rho and anisotropy from ind
    l_a = len(fs.anis)
    l_g = len(fs.gammas['MPs'])
    rho = fs.rhos[ind // (l_a*l_g)]
    anisotropy = fs.anis[ind % (l_a*l_g) // l_g]
    gamma = fs.gammas['MPs'][ind % (l_a*l_g) % l_g]
elif type_computation == 'CO':          #Constant interlayer interaction
    rho = 0
    ind_a = ind // (2*len(fs.gammas['M']))
    ind_l = ind % (2*len(fs.gammas['M']))
    anisotropy = fs.anis[ind_a]
    list_interlayer = ['AA','M']
    place_interlayer = list_interlayer[ind_l//len(fs.gammas['M'])]
    gamma = fs.gammas[place_interlayer][ind_l%len(fs.gammas['M'])]
    moire_pars = {
        'type':'const',
        'place':place_interlayer,
        'theta':fs.thetas,
        }

phys_args = (gamma,rho,anisotropy)

print("Computing Moire: ",moire_pars)
print("Physical parameters are gamma: ","{:.4f}".format(gamma),", rho: ","{:.4f}".format(rho),", anisotropy: ","{:.4f}".format(anisotropy))

#Check if Phi already computed
Phi_fn = fs.get_Phi_fn(moire_pars,machine,rescaled)
if 1:#not Path(Phi_fn).is_file():
    print("Computing interlayer coupling...")
    fs.Moire(moire_pars,machine,rescaled)
#Try a couple of times to load Phi since sometimes it does not work
Phi,a1_m,a2_m = fs.load_Moire(Phi_fn,moire_pars,machine)
fs.plot_Phi(Phi,a1_m,a2_m)
exit()
#######
gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)
grid_pts = (gridx,gridy)

print("Moire lattice vectors: |a_1|=",np.linalg.norm(a1_m),", |a_2|=",np.linalg.norm(a2_m))
print("Relative angle (deg): ","{:.2f}".format(180/np.pi*np.arccos(np.dot(a1_m/np.linalg.norm(a1_m),a2_m/np.linalg.norm(a2_m)))))
print("Grid size: ",gridx,'x',gridy)

if moire_pars['type'] == 'const':
    print("Constant interlayer: ","{:.5f}".format(Phi.sum()/Phi.shape[0]/Phi.shape[1]))
#Compute Phi over new grid parameters
Phi = fs.reshape_Phi(Phi,gridx,gridy)

if 1 and machine =='loc':
    fs.plot_Phi(Phi,a1_m,a2_m)
    exit()

#Check directories for the results exist
fs.check_directory(moire_pars,grid_pts,gamma,machine)

solution_fn = fs.get_sol_fn(moire_pars,grid_pts,phys_args,machine)
if not Path(solution_fn).is_file():
    print("Computing magnetization...")
    args_minimization = {
            'args_moire':       (Phi,(a1_m,a2_m)),
            'moire_pars':       moire_pars,
            'args_phys':        phys_args,
            'grid':             grid_pts,
            'n_initial_pts':    len(fs.list_ind[type_computation]),                         #three solution initial states, 25 constant initial states and n-25 random states
            'maxiter':          1e4, 
            'machine':          machine, 
            'disp':             machine=='loc',
            'type_comp':        type_computation,
            }
    phi = fs.compute_solution(args_minimization)
    if not machine == 'loc':
        np.save(solution_fn,phi)
else:
    print("Already computed")

tot_time = int(time()-t0)
print("Total time: ",tot_time//60," mins ",tot_time%60," secs")










