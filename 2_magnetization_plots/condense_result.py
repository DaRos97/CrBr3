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
print("Computing with ",input_type," values, moire with ",moire_type," strain of ",moire_pars[moire_type]," and gamma: ",gamma)

Phi_fn = fs.get_Phi_fn(moire_type,moire_pars,machine)
a1_m,a2_m = np.load(fs.get_AM_fn(moire_type,moire_pars,machine))
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
        ds_name = fn[len(fn)-fn[::-1].index('/'):-4]
        #if ds_name not in f.keys():
        try:
            f.create_dataset(ds_name,data=np.load(filename))
        except:
            print(fn," not working")

print('Time taken: ',time()-t0)



