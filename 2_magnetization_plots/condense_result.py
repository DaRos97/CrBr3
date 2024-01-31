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
max_gridsize = 200
t0 = time()
machine = fs.get_machine(os.getcwd())

ind = int(sys.argv[1])      #one index every 225 for 15x15 PD
moire_type,moire_pars = fs.get_moire_pars(ind)
print("Condensing PD for Moire with ",moire_type," strain of args ",moire_pars[moire_type])

Phi_fn = fs.get_Phi_fn(moire_type,moire_pars,machine)
Phi = np.load(fs.get_Phi_fn(moire_type,moire_pars,machine))
a1_m,a2_m = np.load(fs.get_AM_fn(moire_type,moire_pars,machine))
gridx,gridy = fs.get_gridsize(max_gridsize,a1_m,a2_m)
#
hdf5_fn = fs.get_hdf5_fn(moire_type,moire_pars,gridx,gridy,machine)

#Open h5py File
with h5py.File(hdf5_fn,'a') as f:
    #List elements in directory
    moire_dn = fs.get_moire_dn(moire_type,moire_pars,gridx,gridy,machine)
    for element in Path(moire_dn).iterdir():
        print(element)
        gamma_dn = str(element)
        if gamma_dn[len(gamma_dn)-gamma_dn[::-1].index('/'):-7]=='gamma':   #-7 fixed by the fact that gamma is saved .4f
            gamma = gamma_dn[-6:]                                    #also here
            if gamma not in f.keys():
                f.create_group(gamma)
            for file in Path(gamma_dn+'/').iterdir():
                print(file)
                sol = str(file)
                dataset_name = gamma+'/'+sol[len(sol)-sol[::-1].index('/')+4:-4]
                if sol[len(sol)-sol[::-1].index('/'):len(sol)-sol[::-1].index('/')+3]=='sol' and dataset_name not in f.keys():
                    f.create_dataset(dataset_name,data=np.load(sol))

#Compute pd figures
fs.compute_PDs(moire_type,moire_pars,gridx,gridy,machine)

print('Time taken: ',time()-t0)



