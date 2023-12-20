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
dic_max_g = {
        334:600, 81:20, 166:20,
        210: 11, 231: 11, 252: 16,
        103: 101, 106: 25, 207: 31,
        246: 171, 263: 251, 266:251,
        290: 201, 312: 401, 350: 401,
        317: 21, 337: 201, 358: 301,
        }
max_g = dic_max_g[s]
#Open h5py File
with h5py.File(fs.name_dir_phi()[:-1]+'.hdf5','r') as f:
    for i in range(max_g):
        parameters = values[i*pts_array**2+s]
        gamma, alpha, beta = parameters
        filename = fs.name_phi(parameters)
        ds_name = filename[len(filename)-filename[::-1].index('/'):-4]
        if ds_name in f.keys():
            phi = np.copy(f[ds_name])
            M.append(np.array([gamma,fs.compute_total_magnetization(phi)]))

#Correct
if s == 266:
    del M[33]
    del M[27]
    del M[26]
    del M[23]
    del M[22]
    del M[17]
    del M[13]
if s == 246:
    del M[28]
    del M[18]
    del M[17]
    del M[14]
if s == 290:
    del M[146]
    del M[72]
    for i in range(6):
        del M[25-i]
    del M[18]
    del M[17]

M[0] = abs(M[0])

M = np.array(M) 
fig = plt.figure(figsize=(20,20))
plt.plot(M[:,0],M[:,1],'r*-')
plt.xlabel(r'$\gamma$',size=s_)
plt.ylabel(r'$M$',size=s_)
plt.title("alpha/(1+alpha) = "+"{:.4f}".format(alpha/(1+alpha))+",  beta/(1+beta) = "+"{:.4f}".format(beta/(1+beta)),size=s_)
plt.show()
