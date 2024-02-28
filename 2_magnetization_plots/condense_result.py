import numpy as np
import functions as fs
import os,sys
import h5py
from pathlib import Path

"""
Remember to adjust max_gridsze.
For each moire dir create a new hdf5, which will contain gamma as dir and (rho,ani) as dataset.
"""

max_grid = 300

machine = fs.get_machine(os.getcwd())

type_computation = 'PD' if len(sys.argv)<3 else sys.argv[2]

ind = int(sys.argv[1])      #one index every 225 for 15x15 PD -> like this sys.argv[1] from 0 to 11
if type_computation == 'CO':
    rho = 0
    ind_a = ind // (2)
    ind_l = ind % (2)
    anisotropy = fs.anis[ind_a]
    #Two cases: AA and M
    list_interlayer = ['AA','M']
    place_interlayer = list_interlayer[ind_l]
    moire_type = 'const'
    moire_pars = {}
    moire_pars[moire_type] = {'place':place_interlayer,}
    moire_pars['theta'] = 0.
elif type_computation == 'DB':
    ggg = [100,200,300,400,500]
    avav = [0,1,2,3,4]
    max_grid = ggg[ind // (5)]
    AV = avav[ind % (5)]
    rho = 1.4
    anisotropy = 0.0709
    moire_type,moire_pars = fs.get_moire_pars(0)
elif type_computation == 'PD':
    moire_type,moire_pars = fs.get_moire_pars(0)        #3% biaxial

Phi_fn = fs.get_Phi_fn(moire_type,moire_pars,machine)
Phi = np.load(fs.get_Phi_fn(moire_type,moire_pars,machine))
a1_m,a2_m = np.load(fs.get_AM_fn(moire_type,moire_pars,machine))
gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)
grid_pts = (gridx,gridy)
print("Condensing PD for Moire with ",moire_type," strain of args ",moire_pars[moire_type])
print("Moire lattice vectors: |a_1|=",np.linalg.norm(a1_m),", |a_2|=",np.linalg.norm(a2_m))
print("Relative angle (deg): ",180/np.pi*np.arccos(np.dot(a1_m/np.linalg.norm(a1_m),a2_m/np.linalg.norm(a2_m))))
print("Constant part of interlayer potential: ",Phi.sum()/Phi.shape[0]/Phi.shape[1]," meV")
print("Grid size: ",gridx,'x',gridy)
#
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

