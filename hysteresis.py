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
#Upload initial state
initial_state = inputs.dic_in_state[int(sys.argv[1])]
filename_hys = fs.name_hys(initial_state,cluster)
try:    #parameters of initial state
    with h5py.File(filename_hys,'r') as f:
        parameters = np.copy(f['parameters'])
except:
    parameters = fs.compute_parameters()[inputs.dic_initial_states[initial_state]]
    with h5py.File(filename_hys,'a') as f:
        f.create_dataset('parameters',data=np.array(parameters))
print("initial state of hysteresis cycle at pars: ",parameters)
try:    #Initial state
    filename_phi = fs.name_phi(parameters,cluster)
    if not cluster:
        f = h5py.File(fs.name_dir_phi(cluster)[:-1]+'.hdf5','r')   #same name as folder but .hdf5
        ds_name = filename_phi[len(filename_phi)-filename_phi[::-1].index('/'):-4]
        phi_0 = np.copy(f[ds_name])
        f.close()
    else:
        phi_0 = np.load(filename_phi)
except:
    print("Initial state not computed, abort")
    exit()
try:    #list_gamma
    with h5py.File(filename_hys,'r') as f:
        list_gamma = np.copy(f['list_gamma'])
    print("Extracted gamma")
except:
    list_gamma = list(np.linspace(0,inputs.limit_gamma,inputs.steps_gamma//2,endpoint=False)) + list(np.linspace(inputs.limit_gamma,-inputs.limit_gamma,inputs.steps_gamma,endpoint=False)) + list(np.linspace(-inputs.limit_gamma,inputs.limit_gamma,inputs.steps_gamma+1))
    with h5py.File(filename_hys,'a') as f:
        f.create_dataset('list_gamma',data=np.array(list_gamma))
    print("Computed gamma")
#
result_phases = np.zeros((len(list_gamma),*phi_0.shape))
result_phases[0] = phi_0
args_hysteresis = {
        'noise':1e-1,
        'learn_rate':-0.01,
        'maxiter':1e6, 
        'disp': not cluster,
        }
for i_g in range(len(list_gamma)):
    try:    #result phase
        with h5py.File(filename_hys,'r') as f:
            result_phases[i_g] = np.copy(f['result_phases/'+str(i_g)])
    except:
        if i_g>0:
            pars = (list_gamma[i_g],*parameters[1:])
            result_phases[i_g] = fs.hysteresis_minimization(Phi,pars,result_phases[i_g-1],args_hysteresis)
        if cluster:
            with h5py.File(filename_hys,'a') as f:
                result_phases_group = f.require_group('result_phases')
                result_phases_group.create_dataset(str(i_g),data=result_phases[i_g])
print("Computed/extracted phases")
#
try:    #Energy and magnetization
    with h5py.File(filename_hys,'r') as f:
        Energy = np.copy(f['Energy'])
        Magnetization = np.copy(f['Magnetization'])
    print("Extracted properties")
except:
    Energy = np.zeros(len(list_gamma))
    Magnetization = np.zeros(len(list_gamma))
    for i_g in range(len(list_gamma)):
        pars = (list_gamma[i_g],*parameters[1:])
        d_phi = (fs.compute_derivatives(result_phases[i_g][0],1),fs.compute_derivatives(result_phases[i_g][1],1))
        Energy[i_g] = fs.compute_energy(result_phases[i_g],Phi,pars,d_phi)
        Magnetization[i_g] = fs.compute_total_magnetization(result_phases[i_g])
    #Save
    with h5py.File(filename_hys,'a') as f:
        f.create_dataset('Energy',data=Energy)
        f.create_dataset('Magnetization',data=Magnetization)
    print("Computed properties")









