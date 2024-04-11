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
#            ['b',400,0.01,1.4,0.02,10,76],    #   
#            ['b',400,0.01,1.4,0.01,10,76],    #   

#            ['b',400,0.05,5,0.02,10,76],    #   
#            ['b',400,0.01,5,0.02,10,76],    #   

            ['b',300,0.01   ,1.4      ,0.01,10,76],    #   
#            ['b',300,0.01   ,1.4      ,0.03,10,76],    #   
            ['b',300,0.01   ,10      ,0.01,10,76],    #   
            ['b',300,0.01   ,10      ,0.03,10,76],    #   
            ['b',300,0.05   ,1.4      ,0.01,10,76],    #   
            ['b',300,0.05   ,1.4      ,0.03,10,76],    #   
#            ['b',300,0.05   ,10      ,0.01,10,76],    #   
            ['b',300,0.05   ,10      ,0.03,10,76],    #   
#            ['b',300,0.01   ,5      ,0.03,10,76],    #   
#            ['b',300,0.005  ,1.4    ,0.03,10,76],    #   
            ]
list_par += [
#        ['u',900,0.01,1.4,0.02,0,17,76],
#        ['u',900,0.05,1.4,0.02,0,17,76],
#        ['u',900,0.03,5,0.02,0,17,76],

#        ['u',500,0.03,10,0.01,0,17,76],
#        ['u',500,0.02,10,0.01,0,17,76],
#        ['u',500,0.01,10,0.01,0,17,76],

#        ['u',400,0.05,5,0.02,1,17,76],
        ]
mps = []
par_fn = []
for MP1_par in list_par:
    MP1_fnn = 'Data/MPs/'
    for i in MP1_par:
        if not i==300 and not i==500:
            MP1_fnn += str(i)+'_'
    MP1_fn = MP1_fnn[:-1]+'.npy'
    MP1_fn2 = MP1_fnn[:-1]+'.txt'
    par_fn.append(MP1_fn2)
    if Path(MP1_fn).is_file():
        MP1 = np.load(MP1_fn)
    else:
        MP1 = pfs.import_theory_data(MP1_par,MP1_fn)
    mps.append(np.array(MP1))

normal = []
modified = []
fac = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#fac = np.ones(10)
for i_ in range(len(mps)):
    normal.append((mps[i_][:,1]-np.min(mps[i_][:,1]))/(np.max(mps[i_][:mps[i_].shape[0]//2,1])-np.min(mps[i_][:,1])))
    modified.append([fac[i_]*normal[-1][i]**(2) for i in range(len(normal[-1]))])
    X = mps[i_][:,0]
    Y = modified[-1]
    res = np.zeros((len(X),2))
    res[:,0] = X
    res[:,1] = Y
    np.savetxt(par_fn[i_],res)
data = modified if squared else normal
################################################################################################################################
################################################################################################################################
################################################################################################################################
#PLOT
if 1:
    fig,axs = plt.subplots(ncols=3,nrows=1,
#            height_ratios=[0.02,1,0.7],
#            width_ratios=[1,1,1,0.1]
                )
    fig.set_size_inches(14,7)
    #Big subplot
    gs = axs[0].get_gridspec()
    for i in range(1):
        for ax in axs:
            ax.remove()
else:
    fig,axs = plt.subplots(ncols=2,nrows=2)
    fig.set_size_inches(8,8)
    gs = axs[0,0].get_gridspec()
    for i in range(2):
        for ax in axs[:,i]:
            ax.remove()

#Theory-Exp plot
ylabel = r'$(\Delta M_z)^2$' if squared else r'$\Delta M_z$'
if 0:
#    ax_big = fig.add_subplot(gs[2,:3])
    ax_big = fig.add_subplot(gs[:,:])
    pfs.theory_exp_plot(ax_big,data,mps,list_par,data1,ylabel)

if 0:#Text on plot
    yyy = 0.69
    ax_big.text(0.035,yyy,r'$(I)$',size=20)
    ax_big.text(0.43,yyy,r'$(II)$',size=20)
    ax_big.text(0.88,yyy,r'$(III)$',size=20)
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

if 1:
    print("Saving picture phases")
    for n in [0,13,30,40,60]:
        np.save('Data/sols/data_ivo_'+"{:.2f}".format(n*2/100)+'T.npy',sol[n])
    exit()


if 0:   #Colored patches to indicate shaded area
    arr_style = patches.ArrowStyle("Fancy", head_length=7, head_width=7, tail_width=0.1)
    XX = 0.305
    lll = 0.05
    YY = 1
    YYarr = YY-0.02
    lenrect = 0.02
    xrect1 = 0.32
    xrect2 = 0.66
    ax = fig.add_subplot(gs[:,:3])
    ax.set_axis_off()
    rectangle = patches.Rectangle((xrect1,YY+0.03),lenrect,-0.619,edgecolor='none',facecolor=pfs.shadeAA,
            alpha=0.3)
    ax.add_patch(rectangle)
    rectangle = patches.Rectangle((xrect2,YY+0.03),lenrect,-0.619,edgecolor='none',facecolor=pfs.shadeM,
            alpha=0.3)
    ax.add_patch(rectangle)
    arrow = patches.FancyArrowPatch((XX,YYarr),(XX+lll,YYarr),arrowstyle=arr_style, color='k')
    ax.add_patch(arrow)
    ax.text(0.32,1.01,r'$B_\bot^{AA}$',size=20)
    arrow = patches.FancyArrowPatch((XX+xrect2-xrect1,YYarr),(XX+xrect2-xrect1+lll,YYarr),arrowstyle=arr_style, color='k')
    ax.add_patch(arrow)
    ax.text(0.66,1.01,r'$B_\bot^M$',size=20)

    ax.text(0.151,YY,r'$(I)$',size=20)
    ax.text(0.481,YY,r'$(II)$',size=20)
    ax.text(0.814,YY,r'$(III)$',size=20)

#Single Hexagons
for i in range(2):
    ax = fig.add_subplot(gs[0,i])
#    ax = fig.add_subplot(gs[:2,:2])
    n = 10
    ax.imshow(sol[n][i])
    continue
    cof, a, b = pfs.plot_hexagon(ax,i,sol,
            moire=0,
            background=1,
            panels=0)
    if i==0:
        min_Phi = a
        max_Phi = b

if 0:#Colorbar
    ax = fig.add_subplot(gs[1,2:])
    pfs.colorbar_v(fig,ax,cof,min_Phi,max_Phi)

    fig.tight_layout()
    plt.subplots_adjust(
            wspace=0.05,
            hspace=0.075,
            )

if 0:
    plt.savefig('/home/dario/Desktop/figs_crbr/picture6.pdf')
    print("Saved")
    exit()
else:
    plt.show()








