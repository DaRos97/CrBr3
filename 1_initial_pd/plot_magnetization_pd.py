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

Order_ds = 'Magnetization_'+"{:.5f}".format(values[i*pts_array**2][0])
#Open h5py File
with h5py.File(fs.name_dir_phi(cluster)[:-1]+'.hdf5','a') as f:
    try:
        Order = np.copy(f[Order_ds])
        if cluster=='loc':
            if input("Use computed order ds? (Y/n)")=='n':
                a = b
    except:
        print("Computing order")
        Order = np.zeros((pts_array,pts_array))
        for j in range(pts_array):
            for k in range(pts_array):
                parameters = values[i*pts_array**2+j*pts_array+k]
                gamma, alpha, beta = parameters
                E0 = -beta-alpha*P0-2*gamma
                filename = fs.name_phi(parameters)
                ds_name = filename[len(filename)-filename[::-1].index('/'):-4]
                try:
                    phi = f[ds_name]
                    Order[j,k] = fs.compute_total_magnetization(phi)
                except:
                    Order[j,k] = np.nan
        if Order_ds in f.keys():
            del f[Order_ds]
        f.create_dataset(Order_ds,data=Order)   

if cluster=='loc':
    parameters = values[i*pts_array**2]
    gamma, alpha, beta = parameters
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex": True,})
    s_ = 20
    plt.figure(figsize=(15,10))
    a = np.linspace(0,1,inputs.pts_array,endpoint=False)
    X,Y = np.meshgrid(a,a)
    plt.contourf(Y,X,Order,levels=20)

    #Phase boundaries
    filename_1 = 'Fit_PD_hejazi/l1.npy'
    line1 = np.load(filename_1)
    plt.plot(line1[:,0],line1[:,1],color='r')
    filename_2 = 'Fit_PD_hejazi/l2.npy'
    line2 = np.load(filename_2)
    plt.plot(line2[:,0],line2[:,1],color='r')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel(r"$\alpha/(1+\alpha)$",size=s_)
    plt.ylabel(r"$\beta/(1+\beta)$",size=s_)
    plt.xticks(size=s_)
    plt.yticks(size=s_)
    plt.title("gamma = "+"{:.4f}".format(gamma))
    plt.show()

