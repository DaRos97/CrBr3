import numpy as np
import functions as fs
import inputs
import sys, os
from pathlib import Path

cluster = False if os.getcwd()[6:11]=='dario' else True
#Parameters in name of solution
args_general = inputs.args_general
pts_array,grid,pts_per_fit,learn_rate_0,A_M = args_general
#Create result directories if they do not exist
fs.check_directories(cluster)
#Parameters of Moire lattice
print("qm: ",4*np.pi/np.sqrt(3)/A_M)
values = fs.compute_grid_pd()
alpha,beta = np.reshape(values,(pts_array**2,2))[int(sys.argv[1])]
print("alpha: ",alpha," beta: ",beta)
print("alpha/1+alpha: ",alpha/(1+alpha)," beta/1+beta: ",beta/(1+beta))

#Check if Phi already computed
filename_Phi = fs.name_Phi(cluster)
try:    
    Phi = np.load(filename_Phi)
except FileNotFoundError:
    print("Computing interlayer coupling...")
    Phi = fs.compute_interlayer()
    #
    np.save(filename_Phi,Phi)

#Check if phi already computed
filename_phi = fs.name_phi(alpha,beta,cluster)
try:
    phi = np.load(filename_phi)
    a = sys.argv[2]
except :
    print("Computing magnetization...")
    args_minimization = {
            'rand_m':100, 
            'maxiter':1e5, 
            'disp': not cluster,
            }
    phi = fs.compute_magnetization(Phi,alpha,beta,args_minimization)
    #
if cluster:
    np.save(filename_phi,phi)

d_phi = (fs.compute_derivatives(phi[0],1),fs.compute_derivatives(phi[1],1))
print("\nFinal energy: ",fs.compute_energy(phi[0],phi[1],Phi,alpha,beta,d_phi))

if not cluster:
    #Actual plot
    #fs.plot_phis(phi[0],phi[1],grid,'final phi_s and phi_a')
    fs.plot_phis(phi[0],np.cos(phi[0]),'phi_s ans cos_phi_s')
    fs.plot_phis(phi[1],np.cos(phi[1]),'phi_a ans cos_phi_a')
#    fs.test_minimum(phi_s,phi_a,Phi,alpha,beta,grid,A_M)
    fs.plot_magnetization(phi[0],phi[1],Phi)



































