import numpy as np
import functions as fs
import os,sys
import h5py
from pathlib import Path

max_grid = 150
AV = 1

machine = fs.get_machine(os.getcwd())

type_computation = sys.argv[1]

i = int(sys.argv[2])
inds = np.array([0,1,2])+5*i
figname = 'aaa.png'
list_pars = []
for ind in inds:
    if type_computation == 'PD':
        #Moire parameters
        moire_type = 'biaxial'
        moire_pars = {}
        moire_pars['theta'] = 0.0
        moire_pars[moire_type] = {}
        moire_pars[moire_type]['eps'] = 0.03
        #Rho,anisotropy
        rho = "{:.5f}".format(fs.rhos[ind//len(fs.anis)])
        anisotropy = "{:.5f}".format(fs.anis[ind%len(fs.anis)])
        txt_name = r'$\rho$:'+rho+', '+r'$d$:'+anisotropy
    elif type_computation == 'CO':
        rho = "{:.5f}".format(0)
        ind_a = ind // (2)
        ind_l = ind % (2)
        anisotropy = "{:.5f}".format(fs.anis[ind_a])
        #Two cases: AA and M
        list_interlayer = ['AA','M']
        place_interlayer = list_interlayer[ind_l]
        moire_type = 'const'
        moire_pars = {}
        moire_pars[moire_type] = {'place':place_interlayer,}
        moire_pars['theta'] = 0.
        txt_name = place_interlayer + r', $\rho$:'+rho+', '+r'$d$:'+anisotropy
    elif type_computation == 'DB':
        ggg = [100,200,300,400,500]
        avav = [0,1,2,3,4]
        max_grid = ggg[ind // (5)]
        AV = avav[ind % (5)]
        rho = "{:.5f}".format(100)
        anisotropy = "{:.5f}".format(0.0709)
        moire_type,moire_pars = fs.get_moire_pars(0)
        txt_name = 'grid='+str(max_grid)+', AV:'+str(AV)

    Phi_fn = fs.get_Phi_fn(moire_type,moire_pars,machine)
    Phi,a1_m,a2_m = fs.load_Moire(Phi_fn,fs.get_AM_fn(moire_type,moire_pars,machine))
    #Precision parameters
    gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)
    precision_pars = (gridx,gridy,AV)
    print("Physical parameters are rho: ",rho,", anisotropy: ",anisotropy)
    print("Condensing PD for Moire with ",moire_type," strain of args ",moire_pars[moire_type])
    print("Moire lattice vectors: |a_1|=",np.linalg.norm(a1_m),", |a_2|=",np.linalg.norm(a2_m))
    print("Relative angle (deg): ",180/np.pi*np.arccos(np.dot(a1_m/np.linalg.norm(a1_m),a2_m/np.linalg.norm(a2_m))))
    print("Constant part of interlayer potential: ",Phi.sum()/Phi.shape[0]/Phi.shape[1]," meV")
    print("Grid size: ",gridx,'x',gridy,', average: ',AV)

    list_pars.append((rho,anisotropy,precision_pars,moire_type,moire_pars,txt_name))

fs.compute_MPs_new(list_pars,figname,machine)

