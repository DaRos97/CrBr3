import numpy as np
import functions as fs
import sys

cluster = False
#Parameters of Moire lattice
pts_array = 30
A_M = 20 #related to theta somehow
print("qm: ",4*np.pi/np.sqrt(3)/A_M)
grid = 100
values = fs.compute_grid_pd(pts_array)
alpha,beta = np.reshape(values,(pts_array**2,2))[int(sys.argv[1])]
print("alpha: ",alpha," beta: ",beta)
#Filenames
filename_s,filename_a = fs.name_phi_sa(alpha,beta,grid,A_M,cluster)
filename_Phi = fs.name_Phi(grid,A_M,cluster)

try:    #Check if Phi already computed
    Phi = np.load(filename_Phi)
except FileNotFoundError:
    print("Computing interlayer coupling...")
    Phi = fs.compute_interlayer(grid,A_M)
    #
    np.save(filename_Phi,Phi)

try:    #Check if already computed alpha and beta
    phi_s = np.load(filename_s)
    phi_a = np.load(filename_a)
except FileNotFoundError:
    print("Computing magnetization...")
    args_minimization = {
            'rand_m':100, 
            'maxiter':1e5, 
            'disp': not cluster,
            }
    phi_s,phi_a = fs.compute_magnetization(Phi,alpha,beta,grid,A_M,args_minimization)
    #
if cluster:
    np.save(filename_s,phi_s)
    np.save(filename_a,phi_a)

d_phi = (fs.compute_derivatives(phi_s,grid,A_M,1),fs.compute_derivatives(phi_a,grid,A_M,1))
print("\nFinal energy: ",fs.compute_energy(phi_s,phi_a,Phi,alpha,beta,grid,A_M,d_phi))

if not cluster:
    #Actual plot
    fs.plot_phis(phi_s,phi_a,grid,'final phi_s and phi_a')
#    fs.test_minimum(phi_s,phi_a,Phi,alpha,beta,grid,A_M)
    fs.plot_magnetization(phi_s,phi_a,Phi,grid)



































