import numpy as np
import functions as fs
import sys, os, h5py
from pathlib import Path

machine = fs.get_machine(os.getcwd())

###########################
max_grid = 150
AV = 1
###########################

type_computation = 'CO' if machine=='loc' else sys.argv[2]

pd_size = len(fs.rhos)*len(fs.anis)
if type_computation == 'PD':
    moire_type,moire_pars = fs.get_moire_pars(int(sys.argv[1])//pd_size)        #for 15*15 PD
    gamma,rho,anisotropy = fs.get_phys_pars(int(sys.argv[1])%pd_size)          
elif type_computation == 'MP':
    input_type,moire_type,moire_pars,gamma = fs.get_MP_pars(int(sys.argv[1]))
    rho = fs.rho_phys[input_type]
    anisotropy = fs.d_phys[input_type]
elif type_computation == 'CO':
    input_type = 'DFT'
    rho = fs.rho_phys[input_type]
    anisotropy = fs.d_phys[input_type]
    #Two cases: AA and M
    list_interlayer = ['AA','M']
    place_interlayer = list_interlayer[int(sys.argv[1])//len(fs.gammas['M'])]
    gamma = fs.gammas[place_interlayer][int(sys.argv[1])%len(fs.gammas['M'])]
    moire_type = 'const'
    moire_pars = {}
    moire_pars[moire_type] = {'place':place_interlayer,}
    moire_pars['theta'] = 0.

print("Computing with Moire with ",moire_type," strain of ",moire_pars[moire_type])
print("Physical parameters are gamma: ","{:.4f}".format(gamma),", rho: ","{:.4f}".format(rho),", anisotropy: ","{:.4f}".format(anisotropy))

#Check if Phi already computed
Phi_fn = fs.get_Phi_fn(moire_type,moire_pars,machine)
if not Path(Phi_fn).is_file():
    print("Computing interlayer coupling...")
    args_Moire = (machine=='loc',moire_type,moire_pars)
    fs.Moire(args_Moire)

Phi = np.load(Phi_fn)
a1_m,a2_m = np.load(fs.get_AM_fn(moire_type,moire_pars,machine))
gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)
precision_pars = (gridx,gridy,AV)

print("Moire lattice vectors: |a_1|=",np.linalg.norm(a1_m),", |a_2|=",np.linalg.norm(a2_m))
print("Constant part of interlayer potential: ",Phi.sum()/Phi.shape[0]/Phi.shape[1]," meV")
print("Grid size: ",gridx,gridy)

#Compute Phi over new grid parameters
Phi = fs.reshape_Phi(Phi,gridx,gridy)

#Extract solution
hdf5_fn = fs.get_hdf5_fn(moire_type,moire_pars,precision_pars,machine)

gamma_str = "{:.4f}".format(gamma)
rho_str = "{:.5f}".format(rho)
ani_str = "{:.5f}".format(anisotropy)
with h5py.File(hdf5_fn,'r') as f:
    for k in f.keys():
        gamma_ = k[-6:]            #-6 fixed by the fact that gamma is saved .4f
        if gamma_ == gamma_str:
            for p in f[k].keys():
                rho = p[:7]      #7 fixed by the fact that rho is saved .5f 
                ani = p[-7:]      #7 fixed by the fact that rho is saved .5f 
                if rho==rho_str and ani==ani_str:
                    solution = np.copy(f[k][p])
                    break
mag = fs.compute_magnetization(solution)
print(mag)
title = 'Mag_'+gamma_str+'_'+str(AV)+'_'+"{:.4f}".format(mag)
fs.plot_magnetization(solution,Phi,(a1_m,a2_m),gamma*3/2/0.607,False)
#fs.plot_phis(solution,(a1_m,a2_m),title)


