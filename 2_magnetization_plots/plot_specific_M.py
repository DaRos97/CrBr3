import numpy as np
import functions as fs
import sys, os, h5py
from pathlib import Path

machine = fs.get_machine(os.getcwd())

###########################
max_grid = 100
###########################

type_computation = 'PD' if len(sys.argv)<3 else sys.argv[2]

pd_size = len(fs.rhos)*len(fs.anis)

ind = int(sys.argv[1])

if type_computation == 'PD':            #Phase Diagram type of physical parameters
    moire_type = 'biaxial'
    moire_pars = {
        'biaxial':{
            'eps':fs.epss[0],
            },
        'theta':fs.thetas,
        }
    l_a = len(fs.anis)
    l_g = len(fs.gammas['MPs'])
    rho = fs.rhos[ind // (l_a*l_g)]
    anisotropy = fs.anis[ind % (l_a*l_g) // l_g]
    gamma = fs.gammas['MPs'][ind % (l_a*l_g) % l_g]
elif type_computation == 'MP':
    input_type,moire_type,moire_pars,gamma = fs.get_MP_pars(int(sys.argv[1]),'MPs')
    rho = fs.rho_phys[input_type]
    anisotropy = fs.d_phys[input_type]
    print("Input type: ",input_type)
elif type_computation == 'CO':
    input_type = 'DFT'
    rho = fs.rho_phys[input_type]
    anisotropy = fs.d_phys[input_type]
    #Two cases: AA and M
    list_interlayer = ['AA','M']
    place_interlayer = list_interlayer[int(sys.argv[1])//len(fs.gammas['M'])]
    gamma = fs.gammas[place_interlayer][int(sys.argv[1])%len(fs.gammas['M'])]
    moire_type = 'const'
    moire_pars = {}
    moire_pars[moire_type] = {'place':place_interlayer,}
    moire_pars['theta'] = 0.
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

print("Computing with Moire with ",moire_type," strain of ",moire_pars[moire_type])
print("Physical parameters are gamma: ","{:.4f}".format(gamma),", rho: ","{:.4f}".format(rho),", anisotropy: ","{:.4f}".format(anisotropy))

#Check if Phi already computed
Phi_fn = fs.get_Phi_fn(moire_type,moire_pars,machine)
if not Path(Phi_fn).is_file():
    print("Computing interlayer coupling...")
    args_Moire = (machine=='loc',moire_type,moire_pars)
    fs.Moire(args_Moire)

Phi = np.load(Phi_fn)
a1_m,a2_m = np.load(fs.get_AM_fn(moire_type,moire_pars,machine))
gridx,gridy = fs.get_gridsize(max_grid,a1_m,a2_m)
grid_pts = (gridx,gridy)

print("Moire lattice vectors: |a_1|=",np.linalg.norm(a1_m),", |a_2|=",np.linalg.norm(a2_m))
print("Relative angle (deg): ",180/np.pi*np.arccos(np.dot(a1_m/np.linalg.norm(a1_m),a2_m/np.linalg.norm(a2_m))))
print("Constant part of interlayer potential: ",Phi.sum()/Phi.shape[0]/Phi.shape[1]," meV")
print("Grid size: ",gridx,'x',gridy)

#Compute Phi over new grid parameters
Phi = fs.reshape_Phi(Phi,gridx,gridy)

#Extract solution
hdf5_fn = fs.get_hdf5_fn(moire_type,moire_pars,grid_pts,machine)

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
mag = fs.compute_magnetization(solution)
print("Magnetization: ",mag)
energy = fs.compute_energy(solution,Phi,phys_args,(a1_m,a2_m),fs.get_M_transf(a1_m,a2_m))

tt = r'$\gamma$:'+gamma_str+', '+r'$\rho$:'+rho_str+', '+r'$d$:'+ani_str+', $\epsilon$:'+"{:.4f}".format(moire_pars[moire_type]['eps'])
fn = 'g:'+gamma_str+'_'+'r:'+rho_str+'_'+'d:'+ani_str+'_e:'+"{:.4f}".format(moire_pars[moire_type]['eps'])
fs.plot_magnetization(solution,Phi,(a1_m,a2_m),gamma,title=tt,save_figname=fn)
#fs.plot_phis(solution,(a1_m,a2_m))


