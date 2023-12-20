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

try:
    #Alpha or beta fixed
    w = sys.argv[1]
    #Fix either alpha or beta with index b/w 0 and pts_array
    i = int(sys.argv[2])
except:
    print("Default inputs")
    w = 'a'
    i = 0

end_gamma = 101
step_gamma = 1

print("Computing phase diagram for ",w,": ",str(values[i][-1]))
Order_ds = 'Order_'+w+'_'+str(i)+'_'+str(end_gamma)+'_'+str(step_gamma)
#Open h5py File
with h5py.File(fs.name_dir_phi(cluster)[:-1]+'.hdf5','a') as f:
    try:
        Order = np.copy(f[Order_ds])
        if cluster=='loc':
            if input("Use computed order ds? (Y/n)")=='n':
                a = b
    except:
        print("Computing order")
        Order = np.zeros((pts_gamma,pts_array),dtype=int)
        for j in range(0,end_gamma,step_gamma):
            for k in range(pts_array):
                parameters = values[j*pts_array**2+i*pts_array+k] if w=='a' else values[j*pts_array**2+k*pts_array+i]
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
                    Order[j,k] = col
                except:
                    Order[j,k] = 1000
        if Order_ds in f.keys():
            del f[Order_ds]
        f.create_dataset(Order_ds,data=Order)   


if cluster=='loc':
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex": True,})
    s_ = 20
    plt.figure(figsize=(15,10))
    for j in range(0,end_gamma,step_gamma):
        for k in range(pts_array):
            parameters = values[j*pts_array**2+i*pts_array+k] if w=='a' else values[j*pts_array**2+k*pts_array+i]
            gamma, alpha, beta = parameters
            y_val = alpha/(1+alpha) if w=='b' else beta/(1+beta)
            if Order[j,k] < 10:
                plt.scatter(gamma,y_val,color=cod_col[Order[j,k]])
    plt.xlabel(r"$\gamma$",size=s_)
    y_ax = r"$\alpha/(1+\alpha)$" if w=='b' else r"$\beta/(1+\beta)$"
    plt.ylabel(y_ax,size=s_)
    plt.show()
