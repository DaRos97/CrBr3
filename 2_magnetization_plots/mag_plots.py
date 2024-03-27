import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

list_pars = [
        [
        ['b',300,0.03,5,0.01],
        ['b',300,0.03,5,0.03],
        ['b',300,0.03,5,0.0709],
        ['b',300,0.03,5,0.11],
        ['b',300,0.03,5,0.15],
            ],
        [
        ['u',500,0.03,5,0.01,0],
        ['u',500,0.03,5,0.03,0],
        ['u',500,0.03,5,0.0709,0],
        ['u',500,0.03,5,0.11,0],
        ['u',500,0.03,5,0.15,0],
            ]
        ]

fig,axs = plt.subplots(ncols=2,nrows=1)
fig.set_size_inches(17,8)

####################################################################
for steps in range(2):
    list_par = list_pars[steps]
    mps = []
    for MP1_par in list_par:
        MP1_fnn = 'Data/MPs/'
        for i in MP1_par:
            if not i==300 and not i==500:
                MP1_fnn += str(i)+'_'
        MP1_fn = MP1_fnn[:-1]
        MP1_par.append(MP1_fn)
        if Path(MP1_fn).is_file():
            MP1 = np.load(MP1_fn)
        else:
            import functions as fs
            import h5py,sys
            translations = fs.translations
            eps_str = "{:.4f}".format(MP1_par[2])
            rho_str = "{:.4f}".format(MP1_par[3])
            ani_str = "{:.4f}".format(MP1_par[4])
            if MP1_par[0] == 'u':
                tr_str = "{:.4f}".format(translations[MP1_par[5]]) if not MP1_par[5]==0 else '0'
            if MP1_par[0] == 'b':
                hdf5_par_fn = 'results/hdf5/par_type:biaxial_eps:'+eps_str+'_theta:0.0000_'+str(MP1_par[1])+'x'+str(MP1_par[1])+'.hdf5'
            else:
                hdf5_par_fn = 'results/hdf5/par_type:uniaxial_eps:'+eps_str+'_ni:0_phi:0_tr:'+tr_str+'_theta:0.0000_'+str(MP1_par[1])+'x3.hdf5'
            MP1 = []
            rho_str = "{:.5f}".format(MP1_par[3])
            ani_str = "{:.5f}".format(MP1_par[4])
            with h5py.File(hdf5_par_fn,'r') as f:
                for k in f.keys():
                    if not k[:5] == 'gamma':
                        continue
                    gamma_ = k[6:]           #6 fixed by len(gamma_)
                    for p in f[k].keys():
                        rho_ = p[:p.index('_')]
                        ani_ = p[p.index('_')+1:]
                        if rho_ == rho_str and ani_ == ani_str:
                            MP1.append([float(gamma_),f[k][p][1]])
            mps.append(np.array(MP1))
            np.save(MP1_fn,mps[-1])

    squared = False
    normal = []
    modified = []
    fac_k = 1
    for i in range(len(mps)):
        X = mps[i][:,0]
        normal.append((mps[i][:,1]-np.min(mps[i][:,1]))/(np.max(mps[i][:,1])-np.min(mps[i][:,1])))
        modified.append([fac_k*normal[-1][i]**(2) for i in range(len(normal[-1]))])

    #PLOT
    #Plot
    label_dic = {'b':'Biaxial','u':'Uniaxial'}
    colors = ['b','orange','green','red','purple']
    for i in range(len(normal)):
        X = mps[i][:,0]
        pp = list_par[i]
        label = label_dic[pp[0]]+': '+r'$\epsilon=$'+"{:.1f}".format(pp[2]*100)+'\%, '+r'$\rho=$'+"{:.1f}".format(pp[3])+' meV, '+r'$d=$'+"{:.2f}".format(pp[4])+' meV'
        if 1:   #either M_z or M_z^2
            if squared:
                axs[steps].plot(X,modified[i],label=label)
            else:
                axs[steps].plot(X,normal[i],label=label)
        else:   #Both
            axs[steps].plot(X,normal[i],label=label,color=colors[i])
            axs[steps].plot(X,modified[i],color=colors[i],linestyle='--')
    #Shaded areas
    shadeAA = 'khaki'
    shadeM = 'chartreuse'
    axs[steps].fill_betweenx(np.linspace(0,1,100),0.1,0.3,facecolor=shadeAA, alpha = 0.3)
    axs[steps].fill_betweenx(np.linspace(0,1,100),0.6,0.95,facecolor=shadeM, alpha = 0.3)
    #Axis
    labelsize = 20
    axs[steps].set_xlabel(r'$B[T]$',size=labelsize)
    if steps == 0:
        if squared:
            axs[steps].set_ylabel(r'$\left(\frac{M_z(B)-M_z(B=0)}{M_z(B=2T)-M_z(B=0)}\right)^2$',size=labelsize)
        else:
            axs[steps].set_ylabel(r'$\frac{M_z(B)-M_z(B=0)}{M_z(B=2T)-M_z(B=0)}$',size=labelsize)
    else:
        axs[steps].set_yticklabels([])
    #Limits and fot size
    axs[steps].set_xlim(0,2)
    axs[steps].set_ylim(0,1)
    axs[steps].tick_params(axis='both',which='major',labelsize=labelsize-5)

    axs[steps].legend(fontsize=15,loc='lower right')

fig.tight_layout()
if 1:
    plt.savefig('/home/dario/Desktop/figs_crbr/MP.pdf')
    print("Saved")
    exit()
else:
    plt.show()

