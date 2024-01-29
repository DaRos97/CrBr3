import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
s_ = 20
import functions as fs
import sys,os,h5py
from pathlib import Path

machine = fs.get_machine(os.getcwd())

ind = int(sys.argv[1])*100
input_type,moire_type,moire_pars,gamma = fs.get_parameters(ind)
Phi_fn = fs.get_Phi_fn(moire_type,moire_pars,machine)
with h5py.File(Phi_fn,'r') as f:
    a1_m = np.copy(f['a1_m'])
    a2_m = np.copy(f['a2_m'])
gridx,gridy = fs.get_gridsize(200,a1_m,a2_m)
#
hdf5_fn = fs.get_hdf5_fn(input_type,moire_type,moire_pars,gamma,gridx,gridy,machine)
#Open and read h5py File
with h5py.File(hdf5_fn,'r') as f:
    M = []
    for k in f.keys():
        gamma = float(k)
        M.append([gamma,fs.compute_magnetization(f[k])])

M[0] = abs(M[0])

M = np.array(M) 
fig = plt.figure(figsize=(20,20))
plt.plot(M[:,0],M[:,1],'r*-')
plt.xlabel(r'$\gamma$',size=s_)
plt.ylabel(r'$M$',size=s_)
title = hdf5_fn[:-5]
plt.title(title)
plt.show()
