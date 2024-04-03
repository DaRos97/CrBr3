import numpy as np
import pic_funs as pfs
from pathlib import Path
import sys
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib
# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"


squared = True #if len(sys.argv)<2 else True

###########################################################################
###########################################################################
#Extract experimental data
save1 = []
save2 = []
for i in range(4):
    save1.append('Data/exp_gr'+str(i)+'.npy')
    save2.append('Data/exp_FGT'+str(i)+'.npy')
if all([Path(save1[i]).is_file() for i in range(4)]) and all([Path(save2[i]).is_file() for i in range(4)]):
    data1 = []
    data2 = []
    for i in range(4):
        data1.append(np.load(save1[i]))
        data2.append(np.load(save2[i]))
else:   #extract
    data1,data2 = pfs.extract_exp_data(save1,save2)
############################################################################################
############################################################################################
#Extract numerical data -> input: [type,grid,moire,rho,ani(,i_tr),index_AA,index_M]
list_par = [
            ['b',300,0.01,1.61,0.02,5,38],    #   
            ]
list_par += [
        ['u',500,0.02,1.4,0.02,0,8,37],
        ]
mps = []
for MP1_par in list_par:
    MP1_fnn = 'Data/MPs/'
    for i in MP1_par:
        if not i==300 and not i==500:
            MP1_fnn += str(i)+'_'
    MP1_fn = MP1_fnn[:-1]
    if Path(MP1_fn).is_file():
        MP1 = np.load(MP1_fn)
    else:
        MP1 = pfs.import_theory_data(MP1_par,MP1_fn)
    mps.append(np.array(MP1))

normal = []
modified = []
fac = [1.7,1.7]
#fac = np.ones(10)
for i_ in range(len(mps)):
    normal.append((mps[i_][:,1]-np.min(mps[i_][:,1]))/(np.max(mps[i_][:,1])-np.min(mps[i_][:,1])))
    modified.append([fac[i_]*normal[-1][i]**(2) for i in range(len(normal[-1]))])
data = modified if squared else normal
################################################################################################################################
################################################################################################################################
################################################################################################################################
#PLOT
smaller_axs = 0.4
fig,axs = plt.subplots(ncols=6,nrows=2,height_ratios=[1,0.1],width_ratios=[1,1,1,smaller_axs,smaller_axs,smaller_axs])
fig.set_size_inches(16,6)
#Big subplot
gs = axs[0,0].get_gridspec()
for i in range(6):
    for ax in axs[:2,i]:
        ax.remove()
ax_big = fig.add_subplot(gs[:2,3:])

#Theory-Exp plot
ylabel = r'$(\Delta M_z)^2$' if squared else r'$\Delta M_z$'
pfs.theory_exp_plot(ax_big,data,mps,list_par,data1,ylabel)

########################################################################
########################################################################
#Hexagons
#Extract a solution to show domain walls
sol_par = ['b',300,0.02,5,0.03]
sol_fnn = 'Data/sols/'
for i in sol_par:
    if not i==300 and not i==500:
        sol_fnn += str(i)+'_'
sol_fn = sol_fnn[:-1]
sol_par.append(sol_fn)
if Path(sol_fn).is_file():
    sol = np.load(sol_fn)
else:
    sol = pfs.import_sol(sol_par,sol_fn)

if 1:   #Colored patches to indicate shaded area
    arr_style = patches.ArrowStyle("Fancy", head_length=7, head_width=7, tail_width=0.1)
    XX = 0.25
    lll = 0.1+(0.275-XX)*2
    YY = 0.94
    ax = fig.add_subplot(gs[0,:3])
    ax.set_axis_off()
    rectangle = patches.Rectangle((0.275,YY+0.03),0.1,-0.95,edgecolor='none',facecolor=pfs.shadeAA,alpha=0.3)
    ax.add_patch(rectangle)
    rectangle = patches.Rectangle((0.275+0.35,YY+0.03),0.1,-0.95,edgecolor='none',facecolor=pfs.shadeM,alpha=0.3)
    ax.add_patch(rectangle)
    arrow = patches.FancyArrowPatch((XX,YY),(XX+lll,YY),arrowstyle=arr_style, color='k')
    ax.add_patch(arrow)
    ax.text(0.32,YY+0.055,r'$h_\bot^{AA}$',size=20)
    arrow = patches.FancyArrowPatch((XX+0.35,YY),(XX+0.35+lll,YY),arrowstyle=arr_style, color='k')
    ax.add_patch(arrow)
    ax.text(0.67,YY+0.055,r'$h_\bot^M$',size=20)

    ax.text(0.132,YY-0.02,r'$(I)$',size=20)
    ax.text(0.477,YY-0.02,r'$(II)$',size=20)
    ax.text(0.823,YY-0.02,r'$(III)$',size=20)

#Single Hexagons
for i in range(3):
    ax = fig.add_subplot(gs[0,i])
    cof, a, b = pfs.plot_hexagon(ax,i,sol,moire=True,background=True,panels=True)
    if i==0:
        min_Phi = a
        max_Phi = b

#Colorbar
ax = fig.add_subplot(gs[:2,0:3])
pfs.colorbar(fig,ax,cof,min_Phi,max_Phi)

fig.tight_layout()
plt.subplots_adjust(wspace=0.3)

if 0:
    plt.savefig('/home/dario/Desktop/figs_crbr/picture6.pdf')
    print("Saved")
    exit()
else:
    plt.show()








