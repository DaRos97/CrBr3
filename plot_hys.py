import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
s_ = 20
import functions as fs
import inputs
import sys, os, h5py

#Upload interlayer coupling
filename_Phi = fs.name_Phi(False)
try:    
    Phi = np.load(filename_Phi)
except FileNotFoundError:
    print("Computing interlayer coupling...")
    Phi = fs.compute_interlayer()
    np.save(filename_Phi,Phi)
#
initial_state = inputs.dic_in_state[int(sys.argv[1])]

filename_hys = fs.name_hys(initial_state,False)
try:
    with h5py.File(filename_hys,'r') as f:
        parameters = np.copy(f['parameters'])
        list_gamma = np.copy(f['list_gamma'])
except:
    print("Error: no gamma/parameters")
    exit()
try:    #Energy and magnetization
    with h5py.File(filename_hys,'r') as f:
        Energy = np.copy(f['Energy'])
        Magnetization = np.copy(f['Magnetization'])
except:
    print("Error: no energy/magnetization")
    Energy = np.zeros(len(list_gamma))
    Magnetization = np.zeros(len(list_gamma))
    for i_g in range(len(list_gamma)):
        try:    #result phase
            with h5py.File(filename_hys,'r') as f:
                result_phase = np.copy(f['result_phases/'+str(i_g)])
            pars = (list_gamma[i_g],*parameters[1:])
            d_phi = (fs.compute_derivatives(result_phase[0],1),fs.compute_derivatives(result_phase[1],1))
            Energy[i_g] = fs.compute_energy(result_phase,Phi,pars,d_phi)
            Magnetization[i_g] = fs.compute_magnetization(result_phase)
        except:
            Energy[i_g] = np.nan
            Magnetization[i_g] = np.nan


#Plot
plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.plot(list_gamma,Energy)
plt.xlabel(r'$\gamma$')
plt.ylabel('Energy')
plt.subplot(1,2,2)
plt.plot(list_gamma,Magnetization)
plt.xlabel(r'$\gamma$')
plt.ylabel('Magnetization per site (sum over 2 layers)')
plt.show()
