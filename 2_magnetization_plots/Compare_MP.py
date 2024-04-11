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
    opts, args = getopt.getopt(argv, "g:",["type=","i_q=","i_p=","i_m=","ni="])
    type_computation = 'PDb'
    ind_q = 1 #->Magnetization
    ind_par = 10
    max_grid = 100
    ind_m = 10
    ind_ni = 0
except:
    print("Error in inputs")
    exit()
for opt, arg in opts:
    if opt == '--type':
        type_computation = arg
    if opt == '--i_q':
        ind_q = int(arg)
    if opt == '--i_p':
        ind_par = int(arg)
    if opt == '--i_m':
        ind_m = int(arg)
    if opt == '--ni':
        ind_ni = int(arg)
    if opt == '-g':
        max_grid = int(arg)

list_pars = []
if type_computation[:2]=='PD':
    if type_computation == 'PDb':
        max_grid = 300
#        list_im = [ind_m,]
        list_im = np.arange(5) if ind_m==10 else [ind_m,]
        inds = np.arange(5)+ind_par*5 if not ind_par==10 else np.arange(25)
#        inds = np.arange(5)*5+ind_par  #fixed d
#        inds = [ind_par,]
        moire_pars = {
            'type':'biaxial',
            'eps':0,       
            'theta':fs.thetas,
            }
    elif type_computation == 'PDu':
        list_im = [ind_m,]
        inds = np.arange(5)+ind_par*5 if not ind_par==10 else np.arange(20)
#        inds = np.arange(20)
        moire_pars = {
            'type':'uniaxial',
            'eps':0,
            'ni':fs.nis[ind_ni],
            'phi':0,
            'tr':0,
            'theta':fs.thetas,
            }
    for i_m in list_im:   #moire 0.05,0.04,0.03,0.02,0.01,0.005
        if type_computation == 'PDu':
            ind_m = i_m//len(fs.translations)
            ind_tr = i_m%len(fs.translations)
            moire_pars['eps'] = fs.epss[ind_m]
            moire_pars['tr'] = fs.translations[ind_tr]
        else:
            moire_pars['eps'] = fs.epss[i_m]

        for ind in inds:
            l_a = len(fs.anis)
            rho = fs.rhos[ind // l_a]
            anisotropy = fs.anis[ind % l_a]
            rho_str = "{:.5f}".format(rho)
            anisotropy_str = "{:.5f}".format(anisotropy)
            txt_name = r'$\rho$:'+rho_str+', '+r'$d$:'+anisotropy_str+', $\epsilon$:'+"{:.4f}".format(moire_pars['eps']) 
            if 0:   #grid size in caption 
                txt_name += ', grid: '+str(max_grid)
            if type_computation=='PDu':
                txt_name += ', tr: '+"{:.2f}".format(moire_pars['tr'])
#
            Phi_fn = fs.get_Phi_fn(moire_pars,machine,rescaled)
            Phi,a1_m,a2_m = fs.load_Moire(Phi_fn,moire_pars,machine)
            #Precision parameters
            gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)
            grid_pts = (gridx,gridy)
            print(txt_name)
         
            list_pars.append((rho_str,anisotropy_str,grid_pts,dict(moire_pars),txt_name))

elif type_computation == 'CO':
    max_grid = 100
    rho = "{:.5f}".format(0)
    for ind in range(0,10):
        ind_a = ind // (2)
        ind_l = ind % (2)
        anisotropy = "{:.5f}".format(fs.anis[ind_a])
        #Two cases: AA and M
        list_interlayer = ['AA','M']
        place_interlayer = list_interlayer[ind_l]
        moire_pars = {
            'type':'const',
            'place':place_interlayer,
            'theta':fs.thetas,
            }
        Phi_fn = fs.get_Phi_fn(moire_pars,machine,rescaled=True)
        Phi,a1_m,a2_m = fs.load_Moire(Phi_fn,moire_pars,machine)
        gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)
        grid_pts = (gridx,gridy)
        txt_name = place_interlayer +r', $d$:'+anisotropy
        grid_pts = (gridx,gridy)
        list_pars.append((rho,anisotropy,grid_pts,moire_pars,txt_name))

#fs.compute_compare_MPs(list_pars,figname,machine,0)
figname = 'bbb.png'
fs.compute_compare_MPs(list_pars,figname,machine,ind_q)
#0 for energy, 1 for magnetization

