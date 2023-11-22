import numpy as np
import functions as fs
import sys

cluster = False
#Parameters of Moire lattice
pts_array = 20
A_M = 20 #related to theta somehow
print("qm: ",4*np.pi/np.sqrt(3)/A_M)
grid = 100
values = fs.compute_grid_pd(pts_array)
alpha,beta = np.reshape(values,(pts_array**2,2))[int(sys.argv[1])]
print("alpha: ",alpha," beta: ",beta)
print("alpha/1+alpha: ",alpha/(1+alpha)," beta/1+beta: ",beta/(1+beta))
#Filenames
filename_phi = fs.name_phi(alpha,beta,grid,A_M,cluster)
filename_Phi = fs.name_Phi(grid,A_M,cluster)

try:    #Check if Phi already computed
    Phi = np.load(filename_Phi)
except FileNotFoundError:
    print("Computing interlayer coupling...")
    Phi = fs.compute_interlayer(grid,A_M)
    #
    np.save(filename_Phi,Phi)

try:    #Check if already computed alpha and beta
    phi = np.load(filename_phi)
except FileNotFoundError:
    print("Computing magnetization...")
    args_minimization = {
            'rand_m':1, 
            'maxiter':1e5, 
            'disp': not cluster,
            }
    phi = fs.compute_magnetization(Phi,alpha,beta,grid,A_M,args_minimization)
    #
if cluster:
    np.save(filename_phi,phi)

d_phi = (fs.compute_derivatives(phi[0],grid,A_M,1),fs.compute_derivatives(phi[1],grid,A_M,1))
print("\nFinal energy: ",fs.compute_energy(phi[0],phi[1],Phi,alpha,beta,grid,A_M,d_phi))

if not cluster:
    #Actual plot
    fs.plot_phis(phi[0],phi[1],grid,'final phi_s and phi_a')
#    fs.test_minimum(phi_s,phi_a,Phi,alpha,beta,grid,A_M)
    fs.plot_magnetization(phi[0],phi[1],Phi,grid)



































