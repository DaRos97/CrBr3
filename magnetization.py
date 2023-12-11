import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
s_ = 20
import functions as fs
import inputs,sys,os,h5py
from pathlib import Path

#Parameters in name of solution
args_general = inputs.args_general
pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = args_general

values = fs.compute_parameters()

s = int(sys.argv[1])           #a-b index
M = []
#Open h5py File
with h5py.File(fs.name_dir_phi()[:-1]+'.hdf5','r') as f:
    for i in range(pts_gamma):
        parameters = values[i*pts_array**2+s]
        gamma, alpha, beta = parameters
        filename = fs.name_phi(parameters)
        ds_name = filename[len(filename)-filename[::-1].index('/'):-4]
        if ds_name in f.keys():
            phi = np.copy(f[ds_name])
            M.append(np.array([gamma,fs.compute_total_magnetization(phi)]))

M = np.array(M)
fig = plt.figure(figsize=(20,20))
plt.plot(M[:,0],M[:,1],'b*-')
plt.xlabel(r'$\gamma$',size=s_)
plt.ylabel(r'$M$',size=s_)
plt.title("alpha/(1+alpha) = "+"{:.4f}".format(alpha/(1+alpha))+",  beta/(1+beta) = "+"{:.4f}".format(beta/(1+beta)),size=s_)
plt.show()
