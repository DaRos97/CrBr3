import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
s_ = 20
import functions as fs
import inputs
import sys, os, h5py

initial_state = 't-s'

#Start hysteresis
filename_hys = fs.name_hys(False)
try:    #list_gamma
    with h5py.File(filename_hys,'r') as f:
        list_gamma = np.copy(f['list_gamma'])
except:
    print("Error: no gamma")
    exit()
try:    #Energy and magnetization
    with h5py.File(filename_hys,'r') as f:
        in_group = f.require_group(initial_state)
        Energy = np.copy(in_group['Energy'])
        Magnetization = np.copy(in_group['Magnetization'])
except:
    print("Error: no energy/magnetization")


#Plot
plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.plot(list_gamma,Energy)
plt.subplot(1,2,2)
plt.plot(list_gamma,Magnetization)
plt.show()
