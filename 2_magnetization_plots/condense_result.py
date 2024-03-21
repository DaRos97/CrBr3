import numpy as np
import functions as fs
import sys, os, h5py
from pathlib import Path
import getopt

##############################################################################
machine = fs.get_machine(os.getcwd())
rescaled = True
##############################################################################
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "i:",["type=","i_m=","max_grid="])
    type_computation = 'PDb'
    ind_moire = 0
    max_grid = 300
except:
    print("Error in inputs")
    exit()
for opt, arg in opts:
    if opt == '--type':
        type_computation = arg
    if opt == '--i_m':
        ind_moire = int(arg)
    if opt == '--max_grid':
        max_grid = int(arg)

if type_computation == 'PDb':
    moire_pars = {
        'type':'biaxial',
        'eps':fs.epss[ind_moire],       
        'theta':fs.thetas,
        }
if type_computation == 'PDu':
    ind_m = ind_moire//len(fs.translations)
    ind_tr = ind_moire%len(fs.translations)
    moire_pars = {
        'type':'uniaxial',
        'eps':fs.epss[ind_m],
        'ni':0,
        'phi':0,
        'tr':fs.translations[ind_tr],
        'theta':fs.thetas,
        }
elif type_computation == 'CO':
    rho = 0
    list_interlayer = ['AA','M']
    place_interlayer = list_interlayer[ind_moire]
    moire_pars = {
        'type':'const',
        'place':place_interlayer,
        'theta':fs.thetas,
        }

Phi_fn = fs.get_Phi_fn(moire_pars,machine,rescaled)
Phi,a1_m,a2_m = fs.load_Moire(Phi_fn,moire_pars,machine)
A_M = (a1_m,a2_m)
gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)
grid_pts = (gridx,gridy)
Phi = fs.reshape_Phi(Phi,gridx,gridy)
print("Condensing PD for Moire with ",moire_pars)
print("Moire lattice vectors: |a_1|=",np.linalg.norm(a1_m),", |a_2|=",np.linalg.norm(a2_m))
print("Relative angle (deg): ","{:.2f}".format(180/np.pi*np.arccos(np.dot(a1_m/np.linalg.norm(a1_m),a2_m/np.linalg.norm(a2_m)))))
print("Grid size: ",gridx,'x',gridy)

if 0 and machine=='loc':
    exit()
#
hdf5_par_fn = fs.get_hdf5_par_fn(moire_pars,grid_pts,machine)
hdf5_fn = fs.get_hdf5_fn(moire_pars,grid_pts,machine)
if not (machine=='loc' and Path(hdf5_fn).is_file()):
    #Open h5py File
    with h5py.File(hdf5_fn,'a') as f:
        #List elements in directory
        moire_dn = fs.get_moire_dn(moire_pars,grid_pts,machine)
        for element in Path(moire_dn).iterdir():
            gamma_dn = str(element)
            if gamma_dn[len(gamma_dn)-gamma_dn[::-1].index('/'):len(gamma_dn)-gamma_dn[::-1].index('_')-1]=='gamma':
                gamma_gn = gamma_dn[len(gamma_dn)-gamma_dn[::-1].index('/'):]        #gn==group name
                if gamma_gn not in f.keys():
                    f.create_group(gamma_gn)
                for file in Path(gamma_dn+'/').iterdir():
                    sol = str(file)
                    dataset_name = gamma_gn+'/'+sol[len(sol)-sol[::-1].index('/')+4:-4]
                    if sol[len(sol)-sol[::-1].index('/'):len(sol)-sol[::-1].index('/')+3]=='sol' and dataset_name not in f.keys():
                        f.create_dataset(dataset_name,data=np.load(sol))
    with h5py.File(hdf5_par_fn,'a') as f:
        #List elements in directory
        moire_dn = fs.get_moire_dn(moire_pars,grid_pts,machine)
        for element in Path(moire_dn).iterdir():
            gamma_dn = str(element)
            if gamma_dn[len(gamma_dn)-gamma_dn[::-1].index('/'):len(gamma_dn)-gamma_dn[::-1].index('_')-1]=='gamma':
                gamma_gn = gamma_dn[len(gamma_dn)-gamma_dn[::-1].index('/'):]        #gn==group name
                if gamma_gn not in f.keys():
                    f.create_group(gamma_gn)
                for file in Path(gamma_dn+'/').iterdir():
                    sol = str(file)
                    sol_dat = np.load(sol)
                    dataset_name = gamma_gn+'/'+sol[len(sol)-sol[::-1].index('/')+4:-4]
                    if sol[len(sol)-sol[::-1].index('/'):len(sol)-sol[::-1].index('/')+3]=='sol' and dataset_name not in f.keys():
                        gamma_ = float(gamma_gn[6:])
                        sol_fn = sol[len(sol)-sol[::-1].index('/'):]
                        rho_ = float(sol_fn[sol_fn.index('_')+1:len(sol_fn)-sol_fn[::-1].index('_')-1])
                        ani_ = float(sol_fn[len(sol_fn)-sol_fn[::-1].index('_'):len(sol_fn)-sol_fn[::-1].index('.')-1])
                        phys_args = (gamma_,rho_,ani_)
                        data_temp = np.array([
                            fs.compute_energy(sol_dat,Phi,phys_args,A_M,fs.get_M_transf(A_M[0],A_M[1])),    #Energy
                            fs.compute_magnetization(sol_dat),          #Mz
                            fs.compute_magnetization_x(sol_dat),        #Mx
                            np.cos(sol_dat[:,0,0]).sum(),                               #Mz AA
                            np.sin(sol_dat[:,0,0]).sum(),                               #Mx AA
                            np.cos(sol_dat[:,sol_dat.shape[0]//3,0]).sum(),             #Mz M
                            np.sin(sol_dat[:,sol_dat.shape[0]//3,0]).sum(),             #Mx M
                            ])
                        f.create_dataset(dataset_name,data=data_temp)



































