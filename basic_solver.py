import numpy as np
import functions as fs
import sys

cluster = True
#Parameters of Moire lattice
pts_array = 20
A_M = 100 #related to theta somehow
grid = 500
values = fs.compute_grid_pd(pts_array)
alpha,beta = np.reshape(values,(pts_array**2,2))[int(sys.argv[1])]
#Filenames
filename_s,filename_a = fs.name_phi_sa(alpha,beta,grid,A_M,cluster)
filename_Phi = fs.name_Phi(grid,A_M,cluster)

try:    #Check if Phi already computed
    Phi = np.load(filename_Phi)
except:
    print("Computing interlayer coupling...")
    Phi = fs.compute_interlayer(grid,A_M)
    #
    np.save(filename_Phi,Phi)

try:    #Check if already computed alpha and beta
    phi_s = np.load(filename_s)
    phi_a = np.load(filename_a)
except:
    print("Computing magnetization...")
    args_minimization = {
            'rand_m':200, 
            'maxiter':1e5, 
            'disp': not cluster,
            }
    phi_s,phi_a = fs.compute_magnetization(Phi,alpha,beta,grid,A_M,args_minimization)
    #
    if cluster:
        np.save(filename_s,phi_s)
        np.save(filename_a,phi_a)
    
print("\nFinal energy: ",fs.compute_energy(phi_s,phi_a,Phi,alpha,beta,grid,A_M))


if not cluster:
    #Actual plot
    #fs.plot_phis(phi_s,phi_a,grid)
    fs.test_minimum(phi_s,phi_a,Phi,alpha,beta,grid,A_M)
    fs.plot_magnetization(phi_s,phi_a,Phi,grid)



































