import numpy as np
import functions as fs
import os,sys
import h5py
from pathlib import Path

max_grid = 100

machine = fs.get_machine(os.getcwd())

type_computation = sys.argv[1]
ind_em = 0  #1 M, 0 E

#ind = int(sys.argv[2])
inds = np.array([19,])#,4,15,19])   0-4:0.1, 5-9:1.4, 10-14:10, 15-19:100
#inds = np.arange(20)
figname = 'aaa.png'
list_pars = []
if type_computation == 'PD':
    for i_m in [0,1,2,3,4,5]:   #moire 0.05,0.04,0.03,0.02,0.01,0.005
        for ind in inds:
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
            txt_name = r'$\rho$:'+rho_str+', '+r'$d$:'+anisotropy_str+', $\epsilon$:'+"{:.4f}".format(fs.epss[i_m])
#
            Phi_fn = fs.get_Phi_fn(moire_type,moire_pars,machine)
            Phi,a1_m,a2_m = fs.load_Moire(Phi_fn,fs.get_AM_fn(moire_type,moire_pars,machine))
            #Precision parameters
            gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)
            grid_pts = (gridx,gridy)
            print("Physical parameters are rho: ",rho_str,", anisotropy: ",anisotropy_str)

            list_pars.append((rho_str,anisotropy_str,grid_pts,moire_type,moire_pars,txt_name))
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


print("Condensing PD for Moire with ",moire_type," strain of args ",moire_pars[moire_type])

#fs.compute_compare_MPs(list_pars,figname,machine,0)
fs.compute_compare_MPs(list_pars,figname,machine,1)
#0 for energy, 1 for magnetization

