import numpy as np
import functions as fs
import inputs,sys,os,h5py
from pathlib import Path

cluster = fs.get_machine(os.getcwd())

#Parameters in name of solution
args_general = inputs.args_general
pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = args_general

values = fs.compute_parameters()
cod_col = ['fuchsia','gold','b','cyan','dodgerblue','r','k']
Phi = np.load(fs.name_Phi())
P0 = np.sum(Phi)/Phi.shape[0]**2

i = int(sys.argv[1])           #gamma index

Order_ds = 'Order_'+"{:.5f}".format(values[i*pts_array**2][0])
#Open h5py File
with h5py.File(fs.name_dir_phi(cluster)[:-1]+'.hdf5','a') as f:
    try:
        Order = np.copy(f[Order_ds])
        if cluster=='loc':
            if input("Use computed order ds? (Y/n)")=='n':
                a = b
    except:
        print("Computing order")
        Order = np.zeros((pts_array,pts_array),dtype=int)
        for j in range(pts_array):
            for k in range(pts_array):
                parameters = values[i*pts_array**2+j*pts_array+k]
                gamma, alpha, beta = parameters
                E0 = -beta-alpha*P0-2*gamma
                filename = fs.name_phi(parameters)
                ds_name = filename[len(filename)-filename[::-1].index('/'):-4]
                try:
                    phi = f[ds_name]
                    phi_s = phi[0]
                    phi_a = phi[1]
                    d_phi = (fs.compute_derivatives(phi_s,1),fs.compute_derivatives(phi_a,1))
                    E = fs.compute_energy(phi,Phi,parameters,d_phi)
                    if E-E0 > 1e-4:       #Solution collinear was not tried for some reason
                        col = 0
                    elif abs(E-E0) < 1e-4:  #collinear
                        col = 1
                    elif abs(np.max(np.cos(phi_s))-np.min(np.cos(phi_s))) < 0.2:    #t-s (all possible)
                        #twisted-s seen by considering a nearly constant cos(phi_s)
                        if abs(np.sum(phi_s)/phi_s.shape[0]**2) < 0.2 or abs(np.sum(phi_s)/phi_s.shape[0]**2-2*np.pi) < 0.2:    #t-s2
                            col = 3
#                        elif abs(np.sum(phi_s)/phi_s.shape[0]**2-np.pi) < 0.2 or abs(np.sum(phi_s)/phi_s.shape[0]**2-3*np.pi) < 0.2:
#                            col = 2
                        else:           #t-s1
                            col = 2
                    else:    #twisted-a
                        col = 5
                except:
                    col = 6
                Order[j,k] = col
        if Order_ds in f.keys():
            del f[Order_ds]
        f.create_dataset(Order_ds,data=Order)   

#Corrections
if i == 0:
    Order[11,7] = 2
    Order[13,6] = 2
    Order[14,4] = 2
    Order[15,4] = 2
    Order[15,5] = 2
if i == 1:
    Order[8,8] = 1
    Order[13,6] = 2
    Order[15,4] = 2
if i == 2:
    Order[15,19] = 1
if i == 3:
    Order[15,18] = 1
    Order[15,19] = 1
    Order[9,9] = 1
    Order[14,2] = 2
    Order[16,2] = 5
    Order[19,1] = 5
if i == 4:
    Order[15,18] = 1
    Order[18,2] = 5
if i == 5:
    Order[16,1] = 3
if i == 10:
    Order[7,3] = 3
    Order[4,2] = 1
    Order[10,3] = 3
    Order[11,9] = 1
if i == 20:
    Order[11,5] = 3
    Order[12,5] = 5
    Order[12,10] = 1
if i == 30:
    Order[8,4] = 3
    Order[11,11] = 1
    Order[15,18] = 1
if i == 40:
    Order[8,4] = 1
    Order[9,8] = 1
    Order[13,7] = 3
    Order[15,18] = 1
if i == 50:
    Order[10,9] = 1
    Order[10,7] = 1
    Order[8,5] = 1
    Order[9,4] = 3
    Order[11,6] = 3
    Order[14,9] = 5
if i == 100:
    Order[12,8] = 3
if i == 200:
    Order[14,7] = 3
    Order[15,11] = 3
    Order[16,15] = 5
if i == 300:
    Order[12,9] = 1
    Order[12,10] = 1
    Order[14,1] = 3
    Order[14,2] = 3
    Order[14,6] = 3
    Order[14,8] = 3
    Order[16,13] = 3
if i == 500:
    Order[15,0] = 3
    Order[16,9] = 3
    Order[17,14] = 3

if cluster=='loc':
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex": True,})
    s_ = 20
    plt.figure(figsize=(15,10))
    for j in range(pts_array):
        for k in range(pts_array):
            parameters = values[i*pts_array**2+j*pts_array+k]
            gamma, alpha, beta = parameters
            plt.scatter(alpha/(1+alpha),beta/(1+beta),color=cod_col[Order[j,k]])
    #Phase boundaries
    filename_1 = 'Fit_PD_hejazi/l1.npy'
    line1 = np.load(filename_1)
    plt.plot(line1[:,0],line1[:,1],color='g')
    filename_2 = 'Fit_PD_hejazi/l2.npy'
    line2 = np.load(filename_2)
    plt.plot(line2[:,0],line2[:,1],color='g')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel(r"$\alpha/(1+\alpha)$",size=s_)
    plt.ylabel(r"$\beta/(1+\beta)$",size=s_)
    plt.xticks(size=s_)
    plt.yticks(size=s_)
    plt.title(r"$\gamma$ = "+"{:.4f}".format(gamma),size=s_)
    plt.show()

