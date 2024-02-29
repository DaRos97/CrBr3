import numpy as np
import functions as fs
import os,sys
import h5py
from pathlib import Path

max_grid = 300

machine = fs.get_machine(os.getcwd())

type_computation = sys.argv[1]

#ind = int(sys.argv[2])
i_m = 0
inds = np.array([0,1,2,3,4])+5*0
figname = 'aaa.png'
list_pars = []
for ind in inds:
    if type_computation == 'PD':
        moire_type = 'biaxial'
        moire_pars = {
            'biaxial':{
                'eps':fs.epss[i_m],       
                },
            'theta':fs.thetas,
            }
        l_a = len(fs.anis)
        rho = fs.rhos[ind // l_a]
        anisotropy = fs.anis[ind % l_a]
        rho_str = "{:.5f}".format(rho)
        anisotropy_str = "{:.5f}".format(anisotropy)
        txt_name = r'$\rho$:'+rho_str+', '+r'$d$:'+anisotropy_str
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
        rho = "{:.5f}".format(1.4) if i==0 else  "{:.5f}".format(100)
        anisotropy = "{:.5f}".format(0.0709)
        moire_type,moire_pars = fs.get_moire_pars(0)
        txt_name = 'grid='+str(max_grid)+', AV:'+str(AV)+', rho='+rho

    Phi_fn = fs.get_Phi_fn(moire_type,moire_pars,machine)
    Phi,a1_m,a2_m = fs.load_Moire(Phi_fn,fs.get_AM_fn(moire_type,moire_pars,machine))
    #Precision parameters
    gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)
    grid_pts = (gridx,gridy)
    print("Physical parameters are rho: ",rho_str,", anisotropy: ",anisotropy_str)
    print("Condensing PD for Moire with ",moire_type," strain of args ",moire_pars[moire_type])
    print("Moire lattice vectors: |a_1|=",np.linalg.norm(a1_m),", |a_2|=",np.linalg.norm(a2_m))
    print("Relative angle (deg): ",180/np.pi*np.arccos(np.dot(a1_m/np.linalg.norm(a1_m),a2_m/np.linalg.norm(a2_m))))
    print("Constant part of interlayer potential: ",Phi.sum()/Phi.shape[0]/Phi.shape[1]," meV")
    print("Grid size: ",gridx,'x',gridy)

    list_pars.append((rho_str,anisotropy_str,grid_pts,moire_type,moire_pars,txt_name))

fs.compute_compare_MPs(list_pars,figname,machine)

