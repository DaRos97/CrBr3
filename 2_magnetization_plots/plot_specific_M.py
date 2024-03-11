import numpy as np
import functions as fs
import sys, os, h5py
from pathlib import Path

machine = fs.get_machine(os.getcwd())

###########################
rescaled = True
###########################

type_computation = 'PDu' #if len(sys.argv)<3 else sys.argv[2]

pd_size = len(fs.rhos)*len(fs.anis)

i_m = int(sys.argv[1])
ind = int(sys.argv[2])

if type_computation[:2]=='PD':
    if type_computation == 'PDb':            #Phase Diagram type of physical parameters
        max_grid = 100
        moire_pars = {
            'type':'biaxial',
            'eps':fs.epss[i_m],       
            'theta':fs.thetas,
            }
    elif type_computation == 'PDu':            #Phase Diagram type of physical parameters
        max_grid = 300
        moire_pars = {
            'type':'uniaxial',
            'eps':fs.epss[i_m],
            'ni':0,
            'phi':0,
            'theta':fs.thetas,
            }
    l_a = len(fs.anis)
    l_g = len(fs.gammas['MPs'])
    rho = fs.rhos[ind // (l_a*l_g)]
    anisotropy = fs.anis[ind % (l_a*l_g) // l_g]
    gamma = fs.gammas['MPs'][ind % (l_a*l_g) % l_g]
elif type_computation == 'CO':
    max_grid = 100
    rho = 0
    ind_a = ind // (2*len(fs.gammas['M']))
    ind_l = ind % (2*len(fs.gammas['M']))
    anisotropy = fs.anis[ind_a]
    #Two cases: AA and M
    list_interlayer = ['AA','M']
    place_interlayer = list_interlayer[ind_l//len(fs.gammas['M'])]
    gamma = fs.gammas[place_interlayer][ind_l%len(fs.gammas['M'])]
    moire_pars = {
        'type':'const',
        'place':place_interlayer,
        'theta':fs.thetas,
        }
elif type_computation == 'DB':
    ggg = [100,200,300,400,500]
    avav = [0,1,2,3,4]
    g_pts = len(fs.gammas['MPs'])
    max_grid = ggg[ind // (5*g_pts)]
    AV = avav[ind % (5*g_pts) //g_pts]
    rho = 1.4
    anisotropy = 0.0709
    gamma = fs.gammas['MPs'][ind % (5*g_pts) %g_pts]
    moire_type,moire_pars = fs.get_moire_pars(0)

print("Computing with Moire with ",moire_pars)
print("Physical parameters are gamma: ","{:.4f}".format(gamma),", rho: ","{:.4f}".format(rho),", anisotropy: ","{:.4f}".format(anisotropy))

#Check if Phi already computed
Phi_fn = fs.get_Phi_fn(moire_pars,machine,rescaled)
if not Path(Phi_fn).is_file():
    print("Computing interlayer coupling...")
    fs.Moire(moire_pars,machine,rescaled)

Phi,a1_m,a2_m = fs.load_Moire(Phi_fn,moire_pars,machine)
gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)
grid_pts = (gridx,gridy)

print("Moire lattice vectors: |a_1|=",np.linalg.norm(a1_m),", |a_2|=",np.linalg.norm(a2_m))
print("Relative angle (deg): ","{:.2f}".format(180/np.pi*np.arccos(np.dot(a1_m/np.linalg.norm(a1_m),a2_m/np.linalg.norm(a2_m)))))
print("Grid size: ",gridx,'x',gridy)

#Compute Phi over new grid parameters
Phi = fs.reshape_Phi(Phi,gridx,gridy)

#Extract solution
hdf5_fn = fs.get_hdf5_fn(moire_pars,grid_pts,machine)

phys_args = (gamma,rho,anisotropy)
gamma_str = "{:.4f}".format(gamma)
rho_str = "{:.5f}".format(rho)
ani_str = "{:.5f}".format(anisotropy)
with h5py.File(hdf5_fn,'r') as f:
    for k in f.keys():
        gamma_ = k[6:]           #6 fixed by len(gamma_)
        if gamma_ == gamma_str:
            for p in f[k].keys():
                rho_ = p[:p.index('_')]
                ani_ = p[p.index('_')+1:]
                if rho_==rho_str and ani_==ani_str:
                    solution = np.copy(f[k][p])
                    break
#mag = fs.compute_magnetization(solution)
#print("Magnetization: ",mag)
#energy = fs.compute_energy(solution,Phi,phys_args,(a1_m,a2_m),fs.get_M_transf(a1_m,a2_m))

tt = r'$\gamma$:'+gamma_str+', '+r'$\rho$:'+rho_str+', '+r'$d$:'+ani_str+', $\epsilon$:'+"{:.4f}".format(moire_pars['eps'])
fn = 'g:'+gamma_str+'_'+'r:'+rho_str+'_'+'d:'+ani_str+'_e:'+"{:.4f}".format(moire_pars['eps'])

fs.plot_magnetization(solution,Phi,(a1_m,a2_m),gamma,title=tt,save_figname=fn,machine=machine)
#fs.plot_phis(solution,(a1_m,a2_m))


