import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"


#PLOT
fig, axs = plt.subplots(nrows=1,ncols=3,
            width_ratios=[1,1,0.1]
            )
gs = axs[0].get_gridspec()
fig.set_size_inches(16,9)
#Interlayer background
import functions as fs
a1,a2 = (np.array([1,0])*2,np.array([-1/2,np.sqrt(3)/2])*2)
DFT = np.load('Data/CrBr3_interlayer_original.npy')
Phi = np.load('Data/CrBr3_interlayer_rescaled.npy')
big_Phi = np.zeros((400,400))
Ds = DFT.shape[0]
big_DFT = np.zeros((2*Ds,2*Ds))
for i_ in range(2):
    for j_ in range(2):
        big_Phi[i_*200:(i_+1)*200,j_*200:(j_+1)*200] = Phi
        big_DFT[i_*Ds:(i_+1)*Ds,j_*Ds:(j_+1)*Ds] = DFT
big_DFT = fs.reshape_Phi(big_DFT,big_Phi.shape[0]*2,big_Phi.shape[1]*2)
big_Phi = fs.reshape_Phi(big_Phi,big_Phi.shape[0]*2,big_Phi.shape[1]*2)
min_Phi = min(np.min(big_DFT),np.min(big_Phi))
max_Phi = max(np.max(big_DFT),np.max(big_Phi))
data = [big_DFT,big_Phi]
txt_label=['(a)','(b)']
for step in range(2):
    X,Y = np.meshgrid(np.linspace(-1,1,data[step].shape[0]),np.linspace(-1,1,data[step].shape[0]))
    A1 = X*a1[0] + Y*a2[0]
    A2 = X*a1[1] + Y*a2[1]
    conditions = [
            A1 >= 1,
            A1 <= -1,
            A2 >= 2/np.sqrt(3)-A1/np.sqrt(3),
            A2 >= 2/np.sqrt(3)+A1/np.sqrt(3),
            A2 <= -2/np.sqrt(3)-A1/np.sqrt(3),
            A2 <= -2/np.sqrt(3)+A1/np.sqrt(3),
    ]
    hexagon_mask = np.any(conditions, axis=0)
    empty = np.where(hexagon_mask, np.zeros(data[step].shape), np.nan)
    #
    ax = axs[step]
    ax.set_aspect(1.)
    ax.set_axis_off()
    ax.text(-0.9,1,txt_label[step],fontsize=20,zorder=11)
    #Interlayer background
    cof = ax.contourf(A1,A2,data[step].T,levels=100,cmap='RdBu')
    con = ax.contour(A1,A2,data[step].T,levels=[0,],colors=('k',),linestyles=('-',),linewidths=(0.5,),
                zorder=9)
    ax.contourf(A1,A2,empty,
            levels=np.linspace(0,1,2),vmin=0,vmax=1e-3,
            cmap='Greys',
            extend='both',
            zorder=10)
    #Limits
    ax.set_xlim(-1.05,1.05)
    ax.set_ylim(-1.2,1.2)
    #UC corners
    y1 = np.linspace(-1/np.sqrt(3),1/np.sqrt(3),100)
    ax.plot(y1*0+1,y1,c='k',zorder=11)
    ax.plot(y1*0-1,y1,c='k',zorder=11)
    x1 = np.linspace(0,1,100)
    ax.plot(x1,1/np.sqrt(3)*2-x1/np.sqrt(3),c='k',zorder=11)
    ax.plot(x1,-1/np.sqrt(3)*2+x1/np.sqrt(3),c='k',zorder=11)
    x1 = np.linspace(-1,0,100)
    ax.plot(x1,1/np.sqrt(3)*2+x1/np.sqrt(3),c='k',zorder=11)
    ax.plot(x1,-1/np.sqrt(3)*2-x1/np.sqrt(3),c='k',zorder=11)
    #High symmetry points
    ps = 70
    fs = 30
    #AA
    ax.scatter(0,0,color='r',s=ps)
    ax.text(-0.06,0.05,'$AA$',size=fs)
    #M
    ax.scatter(2/3,0,color='b',s=ps)
    ax.scatter(-1/3,np.sqrt(3)/3,color='b',s=ps)
    ax.scatter(-1/3,-np.sqrt(3)/3,color='b',s=ps)
    ax.text(2/3,0.05,'$M$',size=fs)
    #M'
    mpc = 'lime'
    ax.scatter(1,0,color=mpc,s=ps,zorder=11)
    ax.scatter(-1/2,np.sqrt(3)/2,color=mpc,s=ps,zorder=11)
    ax.scatter(-1/2,-np.sqrt(3)/2,color=mpc,s=ps,zorder=11)
    ax.text(0.83,-0.15,'$M\'$',size=fs)
    #AB
    ax.scatter(0,2/np.sqrt(3),zorder=11,color='y',s=ps)
    ax.scatter(1,-1/np.sqrt(3),zorder=11,color='y',s=ps)
    ax.scatter(-1,-1/np.sqrt(3),zorder=11,color='y',s=ps)
    ax.text(-0.06,2/np.sqrt(3)-0.15,'$AB$',size=fs,zorder=11)
#Colorbar
axs[2].remove()
ax = fig.add_subplot(gs[:])
ax.set_axis_off()
cbar = fig.colorbar(cof,ax=ax,
        anchor=(0.9,0.3),
        ticks=(min_Phi,0,max_Phi),
        shrink=0.85
        )
cbar.ax.set_yticklabels(["{:.2f}".format(min_Phi),'$0$',"{:.2f}".format(max_Phi)],size=fs)
ax.text(1.2,0.95,r'$J_{int}$',size=fs+5)

fig.tight_layout()
plt.show()
