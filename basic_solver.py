import numpy as np
import functions as fs
import inputs
import sys, os, h5py

cluster = False if os.getcwd()[6:11]=='dario' else True
#Parameters in name of solution
args_general = inputs.args_general
pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = args_general
if cluster:#Create result directories if they do not exist
    fs.check_directories(cluster)
#Parameters of Moire lattice
print("qm: ",4*np.pi/np.sqrt(3)/A_M)
parameters = fs.compute_parameters()[int(sys.argv[1])]
gamma,alpha,beta = parameters
print("alpha: ",alpha," beta: ",beta," gamma: ",gamma)
print("alpha/1+alpha: ",alpha/(1+alpha)," beta/1+beta: ",beta/(1+beta))

#Check if Phi already computed
filename_Phi = fs.name_Phi(cluster)
try:    
    Phi = np.load(filename_Phi)
except FileNotFoundError:
    print("Computing interlayer coupling...")
    Phi = fs.compute_interlayer()
    np.save(filename_Phi,Phi)

try:
    #Check if phi exists
    filename_phi = fs.name_phi(parameters,cluster)
    if not cluster:
        f = h5py.File(fs.name_dir_phi(cluster)[:-1]+'.hdf5','r')   #same name as folder but .hdf5
        ds_name = filename_phi[len(filename_phi)-filename_phi[::-1].index('/'):-4]
        phi = np.copy(f[ds_name])
        f.close()
    else:
        phi = np.load(filename_phi)
    a = sys.argv[2]
except :
    print("Computing magnetization...")
    args_minimization = {
            'rand_m':100, 
            'maxiter':1e5, 
            'disp': not cluster,
            }
    phi = fs.compute_magnetization(Phi,parameters,args_minimization)

d_phi = (fs.compute_derivatives(phi[0],1),fs.compute_derivatives(phi[1],1))
print("\nFinal energy: ",fs.compute_energy(phi[0],phi[1],Phi,parameters,d_phi))

if not cluster:
    #Actual plot
    fs.plot_magnetization(phi[0],phi[1],Phi,parameters)
    if 0:#input("save?(y/N)")=='y':
        dirname = 'results/ivo/'
        filename_s = dirname+'phi_s_'+"{:.8f}".format(alpha)+'_'+"{:.8f}".format(beta)+'.csv'
        filename_a = dirname+'phi_a_'+"{:.8f}".format(alpha)+'_'+"{:.8f}".format(beta)+'.csv'
        np.savetxt(filename_s,phi[0])
        np.savetxt(filename_a,phi[1])



































