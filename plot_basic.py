import numpy as np
import matplotlib.pyplot as plt
import functions as fs

pts_array = 10
grid = 200
pts_per_fit = 31
learn_rate_0 = 1e-2
A_M = 20
args_general = (pts_array,grid,pts_per_fit,learn_rate_0,A_M)

values = fs.compute_grid_pd(pts_array)
cod_col = ['y','k','b','r','gray']
Phi = np.load(fs.name_Phi(grid,A_M))

plt.figure(figsize=(15,10))
for i in range(0,pts_array,1):
    for j in range(0,pts_array,1):
#        continue
        alpha,beta = values[i,j]
        filename_phi = fs.name_phi(alpha,beta,args_general)
        try:
            phi = np.load(filename_phi)
            phi_s = phi[0]
            phi_a = phi[1]
            d_phi = (fs.compute_derivatives(phi_s,args_general,1),fs.compute_derivatives(phi_a,args_general,1))
            E = fs.compute_energy(phi_s,phi_a,Phi,alpha,beta,args_general,d_phi)
            if E+beta > 1e-5:
                col = 'orange'
            elif abs(E+beta) < 1e-3:   #Collinear solution -> const
                col = 'y'
            elif np.sum(np.absolute(phi_s-np.pi)/grid**2) < 1e-1 or (beta==0 and np.sum(np.absolute(d_phi[0]))<1e-5):    
                #twisted-s seen by: either phi_s=pi or phi_s=const at beta=0.
                col = 'b'
            else:    #twisted-a
                col = 'r'
        except FileNotFoundError:
            col = 'k'
        plt.scatter(alpha/(1+alpha),beta/(1+beta),color=col)
filename = 'fit_pol.npy'
pol = np.load(filename)
X = np.linspace(0,1,100)
plt.plot(X,np.poly1d(pol)(X),'b')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

















