import numpy as np
import functions as fs
import os,sys
import h5py
from pathlib import Path

"""
Remember to adjust max_gridsze.
For each moire dir create a new hdf5, which will contain gamma as dir and (rho,ani) as dataset.
"""

max_grid = 200

machine = fs.get_machine(os.getcwd())

type_computation = 'PD' if len(sys.argv)<3 else sys.argv[2]

ind = int(sys.argv[1])      #one index every 225 for 15x15 PD -> like this sys.argv[1] from 0 to 11
if type_computation == 'PD':
    moire_type = 'biaxial'
    moire_pars = {
        'eps':fs.epss[ind],       
        'theta':fs.thetas,
        }
    moire_type = 'uniaxial'
    moire_pars = {
        'eps':fs.epss[ind],
        'ni':0,
        'phi':0,
        'theta':fs.thetas,
        }
elif type_computation == 'CO':
    rho = 0
    #Two cases: AA and M
    list_interlayer = ['AA','M']
    place_interlayer = list_interlayer[ind]
    moire_type = 'const'
    moire_pars = {
        'place':place_interlayer,
        'theta':fs.thetas,
        }
elif type_computation == 'DB':
    ggg = [100,200,300,400,500]
    avav = [0,1,2,3,4]
    max_grid = ggg[ind // (5)]
    AV = avav[ind % (5)]
    rho = 1.4
    anisotropy = 0.0709
    moire_type,moire_pars = fs.get_moire_pars(0)

Phi_fn = fs.get_Phi_fn(moire_type,moire_pars,machine,rescaled=True)
Phi,a1_m,a2_m = fs.load_Moire(Phi_fn,moire_type,moire_pars,machine)
A_M = (a1_m,a2_m)
gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)
grid_pts = (gridx,gridy)
Phi = fs.reshape_Phi(Phi,gridx,gridy)
print("Condensing PD for Moire with ",moire_type," strain of args ",moire_pars)
print("Moire lattice vectors: |a_1|=",np.linalg.norm(a1_m),", |a_2|=",np.linalg.norm(a2_m))
print("Relative angle (deg): ","{:.2f}".format(180/np.pi*np.arccos(np.dot(a1_m/np.linalg.norm(a1_m),a2_m/np.linalg.norm(a2_m)))))
print("Grid size: ",gridx,'x',gridy)
#
hdf5_par_fn = fs.get_hdf5_par_fn(moire_type,moire_pars,grid_pts,machine)
hdf5_fn = fs.get_hdf5_fn(moire_type,moire_pars,grid_pts,machine)
if not (machine=='loc' and Path(hdf5_fn).is_file()):
    #Open h5py File
    with h5py.File(hdf5_fn,'a') as f:
        #List elements in directory
        moire_dn = fs.get_moire_dn(moire_type,moire_pars,grid_pts,machine)
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
        moire_dn = fs.get_moire_dn(moire_type,moire_pars,grid_pts,machine)
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
                        gamma_ = float(gamma_gn[6:])
                        sol_fn = sol[len(sol)-sol[::-1].index('/'):]
                        rho_ = float(sol_fn[sol_fn.index('_')+1:len(sol_fn)-sol_fn[::-1].index('_')-1])
                        ani_ = float(sol_fn[len(sol_fn)-sol_fn[::-1].index('_'):len(sol_fn)-sol_fn[::-1].index('.')-1])
                        phys_args = (gamma_,rho_,ani_)
                        f.create_dataset(dataset_name,data=np.array([fs.compute_energy(np.load(sol),Phi,phys_args,A_M,fs.get_M_transf(A_M[0],A_M[1])),fs.compute_magnetization(np.load(sol))]))



































