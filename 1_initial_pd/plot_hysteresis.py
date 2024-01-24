import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
s_ = 20
import functions as fs
import inputs
import sys, os, h5py

ii = inputs.steps_gamma
list_i_g = [0,  #start
        ii//2,  #inversion
        ii,     #start on other side
        ii+ii//2,   #other inversion
        2*ii,   #start third time
        2*ii+ii//2  #last
        ]

#Upload interlayer coupling
cluster = fs.get_machine(os.getcwd())
filename_Phi = fs.name_Phi(cluster)
try:    
    Phi = np.load(filename_Phi)
except FileNotFoundError:
    print("Computing interlayer coupling...")
    Phi = fs.compute_interlayer()
    np.save(filename_Phi,Phi)
#
initial_state = inputs.dic_in_state[int(sys.argv[1])]

filename_hys = fs.name_hys(initial_state,cluster)
try:
    with h5py.File(filename_hys,'r') as f:
        parameters = np.copy(f['parameters'])
        list_gamma = np.copy(f['list_gamma'])
except:
    print("Error: no gamma/parameters")
    exit()
try:    #Energy and magnetization
    phases = []
    with h5py.File(filename_hys,'r') as f:
        Energy = np.copy(f['Energy'])
        Magnetization = np.copy(f['Magnetization'])
        for i_g in list_i_g:
            phases.append(np.copy(f['result_phases/'+str(i_g)]))
except:
    print("Error: no energy/magnetization")
    with h5py.File(filename_hys,'r') as f:
        max_i_g = len(f['result_phases'].keys())
    for i in list_i_g:
        if i>max_i_g:
            list_i_g = list_i_g[:list_i_g.index(i)]
            list_i_g.append(max_i_g-1)
            max_i_g = 1e20
    phases = []
    Energy = np.zeros(len(list_gamma))
    Magnetization = np.zeros(len(list_gamma))
    for i_g in range(len(list_gamma)):
        try:    #result phase
            with h5py.File(filename_hys,'r') as f:
                result_phase = np.copy(f['result_phases/'+str(i_g)])
            if i_g in list_i_g:
                phases.append(result_phase)
            pars = (list_gamma[i_g],*parameters[1:])
            d_phi = (fs.compute_derivatives(result_phase[0],1),fs.compute_derivatives(result_phase[1],1))
            Energy[i_g] = fs.compute_energy(result_phase,Phi,pars,d_phi)
            Magnetization[i_g] = fs.compute_total_magnetization(result_phase)
        except:
            Energy[i_g] = np.nan
            Magnetization[i_g] = np.nan
#Magnetization patterns
for i in range(len(phases)):
    pars = (list_gamma[list_i_g[i]],*parameters[1:])
    fs.plot_magnetization(phases[i],Phi,pars,True,str(i))
    if i == 0:
        os.system('xdg-open temp0.png')

#
g = np.diff(list_gamma)
e = np.diff(Energy)
m = np.diff(Magnetization)

pos_g = list_gamma[:-1] + g/2
pos_e = Energy[:-1] + e/2
pos_m = Magnetization[:-1] + m/2

norm_1 = np.sqrt(g**2+e**2)
norm_2 = np.sqrt(g**2+m**2)

#E
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(1,2,1)
ax.plot(list_gamma[:ii//2],Energy[:ii//2],marker='o',color='r')
ax.plot(list_gamma[ii//2:ii//2+ii],Energy[ii//2:ii//2+ii],marker='*',color='b')
ax.plot(list_gamma[ii//2+ii:],Energy[ii//2+ii:],marker='^',color='g')
ax.quiver(pos_g,pos_e,g/norm_1,e/norm_1,angles='xy',zorder=5,pivot='mid',scale=80,color='k')
ax.set_xlabel(r'$\gamma$',size=s_)
ax.set_ylabel('Energy',size=s_)
#M
ax = fig.add_subplot(1,2,2)
ax.plot(list_gamma[:ii//2],Magnetization[:ii//2],marker='o',color='r')
ax.plot(list_gamma[ii//2:ii//2+ii],Magnetization[ii//2:ii//2+ii],marker='*',color='b')
ax.plot(list_gamma[ii//2+ii:],Magnetization[ii//2+ii:],marker='^',color='g')
ax.quiver(pos_g,pos_m,g/norm_2,m/norm_2,angles='xy',zorder=5,pivot='mid',scale=80,color='k')
ax.set_xlabel(r'$\gamma$',size=s_)
ax.set_ylabel('Magnetization',size=s_)
plt.show()

os.system('rm temp*.png')




































