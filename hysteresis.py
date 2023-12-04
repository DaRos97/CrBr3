import numpy as np
import functions as fs
import inputs
import sys, os, h5py


cluster = False if os.getcwd()[6:11]=='dario' else True
#Upload interlayer coupling
filename_Phi = fs.name_Phi(cluster)
try:    
    Phi = np.load(filename_Phi)
except FileNotFoundError:
    print("Computing interlayer coupling...")
    Phi = fs.compute_interlayer()
    np.save(filename_Phi,Phi)
initial_state = inputs.dic_in_state[int(sys.argv[1])]
parameters = fs.compute_parameters()[inputs.dic_initial_states[initial_state]]
try:
    filename_phi = fs.name_phi(parameters,cluster)
    if not cluster:
        f = h5py.File(fs.name_dir_phi(cluster)[:-1]+'.hdf5','r')   #same name as folder but .hdf5
        ds_name = filename_phi[len(filename_phi)-filename_phi[::-1].index('/'):-4]
        phi_0 = np.copy(f[ds_name])
        f.close()
    else:
        phi_0 = np.load(filename_phi)
except:
    print("Computing magnetization...")
    args_minimization = {
            'rand_m':100, 
            'maxiter':1e5, 
            'disp': not cluster,
            }
    phi_0 = fs.compute_magnetization(Phi,parameters,args_minimization)
if 0:
    fs.plot_magnetization(phi_0[0],phi_0[1],Phi,parameters)

#Start hysteresis
filename_hys = fs.name_hys(cluster)
try:    #list_gamma
    with h5py.File(filename_hys,'r') as f:
        list_gamma = np.copy(f['list_gamma'])
    print("Extracted gamma")
except:
    list_gamma = list(np.linspace(0,inputs.limit_gamma,inputs.steps_gamma//2,endpoint=False)) + list(np.linspace(inputs.limit_gamma,-inputs.limit_gamma,inputs.steps_gamma,endpoint=False)) + list(np.linspace(-inputs.limit_gamma,inputs.limit_gamma,inputs.steps_gamma+1,endpoint=False))
    with h5py.File(filename_hys,'a') as f:
        f.create_dataset('list_gamma',data=np.array(list_gamma))
    print("Computed gamma")
#
result_phases = np.zeros((len(list_gamma),*phi_0.shape))
result_phases[0] = phi_0
args_hysteresis = {
        'learn_rate':-0.1,
        'maxiter':1e5, 
        'disp': not cluster,
        }
for i_g in range(1,len(list_gamma)):
    try:    #result phase
        with h5py.File(filename_hys,'r') as f:
            in_group = f.require_group(initial_state)
            result_phases[i_g] = np.copy(in_group['result_phases/'+str(i_g)])
    except:
        gamma = list_gamma[i_g]
        result_phases[i_g] = fs.hysteresis_minimization(Phi,parameters,gamma,result_phases[i_g-1],args_hysteresis)
        with h5py.File(filename_hys,'a') as f:
            in_group = f.require_group(initial_state)
            result_phases_group = in_group.require_group('result_phases')
            result_phases_group.create_dataset(str(i_g),data=result_phases[i_g])
print("Computed/extracted phases")
#
try:    #Energy and magnetization
    with h5py.File(filename_hys,'r') as f:
        in_group = f.require_group(initial_state)
        Energy = np.copy(in_group['Energy'])
        Magnetization = np.copy(in_group['Magnetization'])
    print("Extracted properties")
except:
    Energy = np.zeros(len(list_gamma))
    Magnetization = np.zeros(len(list_gamma))
    for i_g in range(len(list_gamma)):
        gamma = list_gamma[i_g]
        pars = (gamma,parameters[1],parameters[2])
        d_phi = (fs.compute_derivatives(result_phases[i_g][0],1),fs.compute_derivatives(result_phases[i_g][1],1))
        Energy[i_g] = fs.compute_energy(result_phases[i_g][0],result_phases[i_g][1],Phi,pars,d_phi)
        Magnetization[i_g] = fs.compute_magnetization(result_phases[i_g][0],result_phases[i_g][1])
    #Save
    with h5py.File(filename_hys,'a') as f:
        in_group = f.require_group(initial_state)
        in_group.create_dataset('Energy',data=Energy)
        in_group.create_dataset('Magnetization',data=Magnetization)
    print("Computed properties")


if not cluster:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex": True,})
    s_ = 20
    #Plot
    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.plot(list_gamma,Energy)
    plt.subplot(1,2,2)
    plt.plot(list_gamma,Magnetization)
    plt.show()







