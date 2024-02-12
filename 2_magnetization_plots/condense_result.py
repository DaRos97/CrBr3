import numpy as np
import functions as fs
import os,sys
import h5py
from pathlib import Path
from time import time

"""
Remember to adjust max_gridsze.
For each moire dir create a new hdf5, which will contain gamma as dir and (rho,ani) as dataset.
"""

max_gridsize = 400
LR = -1e-1
AV = 2

t0 = time()
machine = fs.get_machine(os.getcwd())

type_computation = 'PD' if machine=='loc' else sys.argv[2]

ind = int(sys.argv[1])      #one index every 225 for 15x15 PD -> like this sys.argv[1] from 0 to 11
moire_type,moire_pars = fs.get_moire_pars(ind)
if type_computation == 'CO':
    input_type = 'DFT'
    rho = fs.rho_phys[input_type]
    anisotropy = fs.d_phys[input_type]
    gamma = fs.gammas[int(sys.argv[1])]
    moire_type = 'none'
    moire_pars = {}
    moire_pars[moire_type] = {'l':0,}

print("Condensing PD for Moire with ",moire_type," strain of args ",moire_pars[moire_type])

Phi_fn = fs.get_Phi_fn(moire_type,moire_pars,machine)
Phi = np.load(fs.get_Phi_fn(moire_type,moire_pars,machine))
a1_m,a2_m = np.load(fs.get_AM_fn(moire_type,moire_pars,machine))
gridx,gridy = fs.get_gridsize(max_gridsize,a1_m,a2_m)
precision_pars = (gridx,gridy,LR,AV)
#
hdf5_fn = fs.get_hdf5_fn(moire_type,moire_pars,precision_pars,machine)

#Open h5py File
with h5py.File(hdf5_fn,'a') as f:
    #List elements in directory
    moire_dn = fs.get_moire_dn(moire_type,moire_pars,precision_pars,machine)
    for element in Path(moire_dn).iterdir():
        gamma_dn = str(element)
        if gamma_dn[len(gamma_dn)-gamma_dn[::-1].index('/'):-7]=='gamma':   #-7 fixed by the fact that gamma is saved .4f
            gamma_gn = gamma_dn[len(gamma_dn)-gamma_dn[::-1].index('/'):]        #gn==group name
            if gamma_gn not in f.keys():
                f.create_group(gamma_gn)
            for file in Path(gamma_dn+'/').iterdir():
                sol = str(file)
                dataset_name = gamma_gn+'/'+sol[len(sol)-sol[::-1].index('/')+4:-4]
                if sol[len(sol)-sol[::-1].index('/'):len(sol)-sol[::-1].index('/')+3]=='sol' and dataset_name not in f.keys():
                    f.create_dataset(dataset_name,data=np.load(sol))

if type_computation == 'PD':
    for gamma in [0.,]:        #can be defined each time
        fs.compute_PDs(moire_type,moire_pars,precision_pars,"{:.4f}".format(gamma),machine)
if type_computation in ['CO','MP']:
    for input_type in ['DFT','exp']:    #can be defined each time
        rho = fs.rho_phys[input_type]
        anisotropy = fs.d_phys[input_type]
        fs.compute_MPs(moire_type,moire_pars,precision_pars,"{:.5f}".format(rho),"{:.5f}".format(anisotropy),machine)

print('Time taken: ',time()-t0)



