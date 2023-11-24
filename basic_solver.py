import numpy as np
import functions as fs
import sys, os
from pathlib import Path

cluster = False
#Parameters in name of solution
pts_array = 10
grid = 200
pts_per_fit = 31
learn_rate_0 = 1e-2
A_M = 20
args_general = (pts_array,grid,pts_per_fit,learn_rate_0,A_M)
if not Path(fs.name_dir(args_general,cluster)).is_dir():
    os.system('mkdir '+fs.name_dir(args_general,cluster))
if not Path(fs.name_dir_Phi(cluster)).is_dir():
    os.system('mkdir '+fs.name_dir_Phi(cluster))
#Parameters of Moire lattice
print("qm: ",4*np.pi/np.sqrt(3)/A_M)
values = fs.compute_grid_pd(pts_array)
alpha,beta = np.reshape(values,(pts_array**2,2))[int(sys.argv[1])]
print("alpha: ",alpha," beta: ",beta)
print("alpha/1+alpha: ",alpha/(1+alpha)," beta/1+beta: ",beta/(1+beta))
#Filenames
filename_phi = fs.name_phi(alpha,beta,args_general,cluster)
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
    a = sys.argv[2]
except :
    print("Computing magnetization...")
    args_minimization = {
            'rand_m':100, 
            'maxiter':1e5, 
            'disp': not cluster,
            }
    phi = fs.compute_magnetization(Phi,alpha,beta,args_general,args_minimization)
    #
if cluster:
    np.save(filename_phi,phi)

d_phi = (fs.compute_derivatives(phi[0],args_general,1),fs.compute_derivatives(phi[1],args_general,1))
print("\nFinal energy: ",fs.compute_energy(phi[0],phi[1],Phi,alpha,beta,args_general,d_phi))

if not cluster:
    #Actual plot
    #fs.plot_phis(phi[0],phi[1],grid,'final phi_s and phi_a')
    fs.plot_phis(phi[0],np.cos(phi[0]),grid,'phi_s ans cos_phi_s')
    fs.plot_phis(phi[1],np.cos(phi[1]),grid,'phi_a ans cos_phi_a')
#    fs.test_minimum(phi_s,phi_a,Phi,alpha,beta,grid,A_M)
    fs.plot_magnetization(phi[0],phi[1],Phi,grid)



































