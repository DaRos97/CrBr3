import numpy as np
import functions as fs
import os,sys
import h5py
from pathlib import Path
from time import time

"""Remember to adjust max_gridsize
"""
max_gridsize = 100
t0 = time()
machine = fs.get_machine(os.getcwd())

ind = int(sys.argv[1])*100  #so each ind takes all gammas of a different parameter of Moire
input_type,moire_type,moire_pars,gamma = fs.get_parameters(ind)
Phi_fn = fs.get_Phi_fn(moire_type,moire_pars,machine)
with h5py.File(Phi_fn,'r') as f:
    a1_m = np.copy(f['a1_m'])
    a2_m = np.copy(f['a2_m'])
gridx,gridy = fs.get_gridsize(max_gridsize,a1_m,a2_m)
#
hdf5_fn = fs.get_hdf5_fn(input_type,moire_type,moire_pars,gamma,gridx,gridy,machine)
#Remove it if it already exists
if 0:#not cluster == 'loc':
    if Path(hdf5_fn).is_file():
        os.system('rm '+hdf5_fn)
#Open h5py File
with h5py.File(hdf5_fn,'a') as f:
    #List elements in directory
    dn_sol = fs.get_sol_dn(input_type,moire_type,moire_pars,gamma,gridx,gridy,machine)
    for filename in Path(dn_sol).iterdir():
        fn = str(filename)
        ds_name = fn[:-4]
        print(ds_name)
        exit()
        #if ds_name not in f.keys():
        try:
            f.create_dataset(ds_name,data=np.load(dn_sol+filename))
        except:
            print(fn)

print('Time taken: ',time()-t0)



