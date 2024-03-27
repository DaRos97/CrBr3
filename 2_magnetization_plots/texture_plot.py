import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def find_color(x):
    """
    x from 0 to 1
    """
    if x < 0.5:
        return (1-x*2,1,0)
    else:
        y = (x-0.5)*2   #from 0 to 1
        return (0+y*153/255,1-y*204/255,y)

def dist_xy(x,y):
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

def inside_UC(a,b,a1_m,a2_m,a12_m):
    x = np.array([a,b])
    if dist_xy(x,a2_m)<np.linalg.norm(x) and dist_xy(x,a2_m)<dist_xy(x,a12_m):
        return a-a2_m[0],b-a2_m[1]
    elif dist_xy(x,a1_m)<np.linalg.norm(x) and dist_xy(x,a1_m)<dist_xy(x,a12_m):
        return a-a1_m[0],b-a1_m[1]
    elif dist_xy(x,a12_m)<np.linalg.norm(x):
        return a-a1_m[0]-a2_m[0],b-a1_m[1]-a2_m[1]
    else:
        return a,b

sol_par = ['b',300,0.02,5,0.01]

#Extract a solution to show domain walls
sol_fnn = 'Data/sols/'
for i in sol_par:
    if not i==300 and not i==500:
        sol_fnn += str(i)+'_'
sol_fn = sol_fnn[:-1]
sol_par.append(sol_fn)
if Path(sol_fn).is_file():
    sol = np.load(sol_fn)
else:
    import functions as fs
    import h5py,sys
    translations = fs.translations
    eps_str = "{:.4f}".format(sol_par[2])
    rho_str = "{:.4f}".format(sol_par[3])
    ani_str = "{:.4f}".format(sol_par[4])
    hdf5_par_fn = 'results/hdf5/type:biaxial_eps:'+eps_str+'_theta:0.0000_'+str(sol_par[1])+'x'+str(sol_par[1])+'.hdf5'
    sol = []
    rho_str = "{:.5f}".format(sol_par[3])
    ani_str = "{:.5f}".format(sol_par[4])
    with h5py.File(hdf5_par_fn,'r') as f:
        for k in f.keys():
            if not k[:5] == 'gamma':
                continue
            gamma_ = k[6:]           #6 fixed by len(gamma_)
            for p in f[k].keys():
                rho_ = p[:p.index('_')]
                ani_ = p[p.index('_')+1:]
                if rho_ == rho_str and ani_ == ani_str:
                    sol.append(np.copy(f[k][p]))
    sol = np.array(sol)
    np.save(sol_fn,sol)

####################################################################################################################
####################################################################################################################
####################################################################################################################
#PLOT
fig, axs = plt.subplots(nrows=2,ncols=4)
fig.set_size_inches(16,9)
#Big subplot
gs = axs[0,0].get_gridspec()
for i in range(4):
    for ax in axs[:2,i]:
        ax.remove()
#Interlayer background
a1,a2 = (np.array([1,0])*2,np.array([-1/2,np.sqrt(3)/2])*2)
X,Y = np.meshgrid(np.linspace(-1,1,400),np.linspace(-1,1,400))
A1 = X*a1[0] + Y*a2[0]
A2 = X*a1[1] + Y*a2[1]
Phi = np.load('Data/CrBr3_interlayer_rescaled.npy')
big_Phi = np.zeros((400,400))
for i_ in range(2):
    for j_ in range(2):
        big_Phi[i_*200:(i_+1)*200,j_*200:(j_+1)*200] = Phi
ind_M = [1,20,99]
#Plotting
for step in range(3):
    for layer in range(2):
        ax = fig.add_subplot(gs[step//2,step*2-(step//2)*3+layer])
        ax.set_aspect(1.)
        ax.set_axis_off()
        #Interlayer background
        cof = ax.contourf(A1,A2,big_Phi.T,levels=100,cmap='RdBu')
        con = ax.contour(A1,A2,big_Phi.T,levels=[0,],colors=('r',),linestyles=('-',),linewidths=(0.5,))
        #UC corners
        y1 = np.linspace(-1/np.sqrt(3),1/np.sqrt(3),100)
        ax.plot(y1*0+1,y1,c='k')
        ax.plot(y1*0-1,y1,c='k')
        x1 = np.linspace(0,1,100)
        ax.plot(x1,1/np.sqrt(3)*2-x1/np.sqrt(3),c='k')
        ax.plot(x1,-1/np.sqrt(3)*2+x1/np.sqrt(3),c='k')
        x1 = np.linspace(-1,0,100)
        ax.plot(x1,1/np.sqrt(3)*2+x1/np.sqrt(3),c='k')
        ax.plot(x1,-1/np.sqrt(3)*2-x1/np.sqrt(3),c='k')
        #Limits
        ax.set_xlim(-1.05,1.05)
        ax.set_ylim(-2/np.sqrt(3)-0.05,2/np.sqrt(3)+0.05)
        #Arrows
        phi = sol[ind_M[step]][layer]
        nspins = 15
        arr_len = 0.07
        if 1:
            for i in range(nspins):
                for j in range(nspins):
                    x_c = i/nspins
                    y_c = j/nspins
                    x = x_c*a1[0] + y_c*a2[0]
                    y = x_c*a1[1] + y_c*a2[1]
                    x,y = inside_UC(x,y,a1,a2,a1+a2)
                    phi_fin = phi[int(x_c*phi.shape[0]),int(y_c*phi.shape[0])]
                    aa = np.copy(phi_fin)
                    if aa < 0:
                        aa += 2*np.pi
                    if aa > np.pi:
                        aa = 2*np.pi-aa
                    bb = aa/np.pi
                    color = find_color(bb)
                    ax.arrow(x - arr_len/2*np.sin(phi_fin),y - arr_len/2*np.cos(phi_fin),arr_len*np.sin(phi_fin), arr_len*np.cos(phi_fin),
                            head_width=0.03,
                            head_length=0.05,
                            color=color,
                            lw=0.5,
                            )
    #square the two hexagons
    ax = fig.add_subplot(gs[step//2,step*2-(step//2)*3:step*2-(step//2)*3+2])
    ax.set_axis_off()
    rectangle = patches.Rectangle((0,0),1,1,edgecolor='k',facecolor='none',linewidth=7)
    ax.add_patch(rectangle)
if 1:
    #Colorbar
    ax = fig.add_subplot(gs[1,0])
    ax.set_axis_off()
    cbar = fig.colorbar(cof,ax=ax,shrink=0.9,location='right',ticks=[np.min(Phi),np.max(Phi)])
    cbar.ax.set_yticklabels(['$FM$','$AFM$'],size=20)

fig.tight_layout()
plt.subplots_adjust(bottom=0.05,top=0.955)

if 0:
    plt.savefig('/home/dario/Desktop/figs_crbr/textures.pdf')
    print("Saved")
    exit()
else:
    plt.show()




