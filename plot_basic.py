import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
s_ = 20
import functions as fs
import inputs,sys,os,h5py
from pathlib import Path

cluster = False if os.getcwd()[6:11]=='dario' else True

#Parameters in name of solution
args_general = inputs.args_general
pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = args_general

values = fs.compute_parameters()
cod_col = ['y','k','b','r','gray']
Phi = np.load(fs.name_Phi())
P0 = np.sum(Phi)/Phi.shape[0]**2

plt.figure(figsize=(15,10))
i = int(sys.argv[1])           #gamma index
#Open h5py File
with h5py.File(fs.name_dir_phi(cluster)[:-1]+'.hdf5','r') as f:
    for j in range(pts_array):
        for k in range(pts_array):
            parameters = values[i*pts_array**2+j*pts_array+k]
            gamma, alpha, beta = parameters
            E0 = -beta+alpha*P0-2*gamma
            filename = fs.name_phi(parameters)
            ds_name = filename[len(filename)-filename[::-1].index('/'):-4]
            try:
                phi = f[ds_name]
                phi_s = phi[0]
                phi_a = phi[1]
                d_phi = (fs.compute_derivatives(phi_s,1),fs.compute_derivatives(phi_a,1))
                E = fs.compute_energy(phi,Phi,parameters,d_phi)
                if E-E0 > 1e-4:       #Solution collinear was not tried for some reason
                    col = 'fuchsia'
                elif abs(E-E0) < 1e-4:
                    col = 'gold'
                elif abs(np.max(np.cos(phi_s))-np.min(np.cos(phi_s))) < 0.3:
                    #twisted-s seen also by considering a nearly constant cos(phi_s)
                    col = 'dodgerblue'
                else:    #twisted-a
                    col = 'r'
            except:
                col = 'k'
            plt.scatter(alpha/(1+alpha),beta/(1+beta),color=col)
#Phase boundaries
filename_1 = 'Fit_PD_hejazi/l1.npy'
line1 = np.load(filename_1)
plt.plot(line1[:,0],line1[:,1],color='b')
filename_2 = 'Fit_PD_hejazi/l2.npy'
line2 = np.load(filename_2)
plt.plot(line2[:,0],line2[:,1],color='b')
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel(r"$\alpha/(1+\alpha)$",size=s_)
plt.ylabel(r"$\beta/(1+\beta)$",size=s_)
plt.xticks(size=s_)
plt.yticks(size=s_)
plt.title("gamma = "+"{:.4f}".format(gamma))
plt.show()

