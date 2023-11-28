import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
s_ = 20
import functions as fs
import sys
import inputs

#Parameters in name of solution
args_general = inputs.args_general
pts_array,grid,pts_per_fit,learn_rate_0,A_M = args_general

values = fs.compute_grid_pd()
cod_col = ['y','k','b','r','gray']
Phi = np.load(fs.name_Phi())

plt.figure(figsize=(15,10))
for i in range(0,pts_array,1):
    for j in range(0,pts_array,1):
        alpha,beta = values[i,j]
        filename_phi = fs.name_phi(alpha,beta)
        try:
            phi = np.load(filename_phi)
            phi_s = phi[0]
            phi_a = phi[1]
            d_phi = (fs.compute_derivatives(phi_s,1),fs.compute_derivatives(phi_a,1))
            E = fs.compute_energy(phi_s,phi_a,Phi,alpha,beta,d_phi)
            if E+beta > 1e-5:       #Solution collinear was not tried for some reason
                col = 'orange'
            elif abs(E+beta) < 1e-3:   #Collinear solution -> const
                col = 'y'
            elif np.sum(np.absolute(phi_s-np.pi)/grid**2) < 1e-1 or (beta==0 and np.sum(np.absolute(d_phi[0]))<1e-5):    
                #twisted-s seen by: either phi_s=pi or phi_s=const at beta=0.
                col = 'b'
            elif abs(np.max(np.cos(phi_s))-np.min(np.cos(phi_s))) < 0.1:
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
plt.show()

















