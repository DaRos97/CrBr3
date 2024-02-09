import numpy as np
import functions as fs
import sys, os, h5py
from pathlib import Path

machine = fs.get_machine(os.getcwd())

type_computation = 'MP' if machine=='loc' else sys.argv[2]

pd_size = len(fs.rhos)*len(fs.anis)
if type_computation == 'PD':
    moire_type,moire_pars = fs.get_moire_pars(int(sys.argv[1])//pd_size)        #for 15*15 PD
    gamma,rho,anisotropy = fs.get_phys_pars(int(sys.argv[1])%pd_size)          
elif type_computation == 'MP':
    input_type,moire_type,moire_pars,gamma = fs.get_MP_pars(int(sys.argv[1]))
    rho = fs.rho_phys[input_type]
    anisotropy = fs.d_phys[input_type]

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
###########################
max_grid = 200
LR = -1e-1
AV = 3
###########################
gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)
precision_pars = (gridx,gridy,LR,AV)

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
        gamma = k[-6:]            #-6 fixed by the fact that gamma is saved .4f
        if gamma == gamma_str:
            for p in f[k].keys():
                rho = p[:7]      #7 fixed by the fact that rho is saved .5f 
                ani = p[-7:]      #7 fixed by the fact that rho is saved .5f 
                if rho==rho_str and ani==ani_str:
                    solution = np.copy(f[k][p])
                    break
fs.plot_magnetization(solution,Phi,(a1_m,a2_m),rho_str+'_'+ani_str)
fs.plot_phis(solution,(a1_m,a2_m),rho_str+'_'+ani_str)


