import numpy as np
import functions as fs
import os,sys
import h5py
from pathlib import Path

max_gridsize = 150
AV = 1

machine = fs.get_machine(os.getcwd())

type_computation = 'MP'

inds = [0,3,16,19]#int(sys.argv[1])
input_type = 'DFT'
list_pars = []
for ind in inds:
    moire_type,moire_pars = fs.get_moire_pars(ind)
    print("Condensing PD for Moire with ",moire_type," strain of args ",moire_pars[moire_type])

    a1_m,a2_m = np.load(fs.get_AM_fn(moire_type,moire_pars,machine))
    gridx,gridy = fs.get_gridsize(max_gridsize,a1_m,a2_m)
    precision_pars = (gridx,gridy,AV)

    txt_name = r'$\epsilon$:'+"{:.2f}".format(moire_pars[moire_type]['eps'])+', '+r'$\nu$:'+"{:.2f}".format(moire_pars[moire_type]['ni'])
    list_pars.append((moire_type,moire_pars,precision_pars,txt_name))
rho = fs.rho_phys[input_type]
anisotropy = fs.d_phys[input_type]
    

fs.compute_MPs_new(list_pars,"{:.5f}".format(rho),"{:.5f}".format(anisotropy),machine)

