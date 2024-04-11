import numpy as np
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


squared = False if len(sys.argv)<2 else True

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
    data1_fn = 'Data/grinjection_fourdevice.dat'
    data2_fn = 'Data/FGTinjection_fourdevice.dat'

    dev1 = []
    dev2 = []
    dev3 = []
    dev4 = []
    with open(data1_fn, 'r') as f:
        d = f.readlines()
        for i in d[1:]:
            a = i.split()
            if len(a) == 8:
                dev1.append([float(a[0]),float(a[1])])
                dev2.append([float(a[2]),float(a[3])])
                dev3.append([float(a[4]),float(a[5])])
                dev4.append([float(a[6]),float(a[7])])
            elif len(a) == 6:
                dev2.append([float(a[0]),float(a[1])])
                dev3.append([float(a[2]),float(a[3])])
                dev4.append([float(a[4]),float(a[5])])
    dev1 = np.array(dev1)
    dev2 = np.array(dev2)
    dev3 = np.array(dev3)
    dev4 = np.array(dev4)
    data1 = [dev1,dev2,dev3,dev4]
    for i in range(4):
        np.save(save1[i],data1[i])
    #
    adev1 = []
    adev2 = []
    adev3 = []
    adev4 = []
    with open(data2_fn, 'r') as f:
        d = f.readlines()
        for i in d[1:]:
            a = i.split()
            if len(a) == 8:
                adev1.append([float(a[0]),float(a[1])])
                adev2.append([float(a[2]),float(a[3])])
                adev3.append([float(a[4]),float(a[5])])
                adev4.append([float(a[6]),float(a[7])])
            elif len(a) == 6:
                adev2.append([float(a[1]),float(a[2])])
                adev3.append([float(a[3]),float(a[4])])
            elif len(a) < 6:
                adev2.append([float(a[0]),float(a[1])])
                adev3.append([float(a[2]),float(a[3])])
    adev1 = np.array(adev1)
    adev2 = np.array(adev2)
    adev3 = np.array(adev3)
    adev4 = np.array(adev4)

    data2 = [adev1,adev2,adev3,adev4]
    for i in range(4):
        np.save(save2[i],data2[i])
############################################################################################
############################################################################################
#Extract numerical data -> input: [type,grid,i_m(,i_tr),i_r,i_a]
####################################################################
excl = [(0,4,0),(0,4,1),
        (1,0,3),(1,0,4),
        (2,0,3),(2,0,4),
        (3,0,0),(3,0,2),(3,0,3),(3,0,4),(3,1,4),
        (4,3,3),(4,3,4),(4,2,2),(4,2,3),(4,2,4),(4,1,2),(4,1,3),(4,1,4),(4,0,2),(4,0,3),(4,0,4)]

list_par = [
#            ['b',300,0.05,1.4,0.02],    #   
            ['b',300,0.01,1.61,0.02,5,38],    #   
#            ['b',300,0.005,100,0.03],    #
            ]
list_par += [
#        ['u',500,0.03,1.4,0.03,0],
#        ['u',500,0.02,1.4,0.03,0,9,46],
#        ['u',500,0.02,10,0.03,0,7,45],
        ['u',500,0.02,1.4,0.02,0,8,37],
        ]
####################################################################
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

normal = []
modified = []
fac = [1.7,1.7]
#fac = np.ones(10)
for i_ in range(len(mps)):
    normal.append((mps[i_][:,1]-np.min(mps[i_][:,1]))/(np.max(mps[i_][:,1])-np.min(mps[i_][:,1])))
    modified.append([fac[i_]*normal[-1][i]**(2) for i in range(len(normal[-1]))])

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
#Plot
colors_mp = ['red','brown','orange','maroon','darkorange','orangered','firebrick','maroon']
label_dic = {'b':'Biaxial','u':'Uniaxial'}
list_labels = []
for i in range(len(normal)):
    X = mps[i][:,0]
    pp = list_par[i]
    list_labels.append(label_dic[pp[0]]+': '+r'$\epsilon=$'+"{:.0f}".format(pp[2]*100)+'\%, '+r'$\rho=$'+"{:.1f}".format(pp[3])+' meV, '+r'$d=$'+"{:.2f}".format(pp[4])+' meV')
    if squared:
        ax_big.plot(X,modified[i],color=colors_mp[i])
        ax_big.scatter(X[pp[-2]],modified[i][pp[-2]],marker='*',color=colors_mp[i])
        ax_big.scatter(X[pp[-1]],modified[i][pp[-1]],marker='*',color=colors_mp[i])
    else:
        ax_big.plot(X,normal[i],color=colors_mp[i])
        ax_big.scatter(X[pp[-2]],normal[i][pp[-2]],marker='*',color=colors_mp[i])
        ax_big.scatter(X[pp[-1]],normal[i][pp[-1]],marker='*',color=colors_mp[i])
lis = np.arange(4)
colors_gr = ['paleturquoise','turquoise','c','darkcyan']
for i in lis:
    x = data1[i][:,0]
    ax_big.plot(x,data1[i][:,1]/data1[i][-1,1],ls='--',
            color=colors_gr[i],
            zorder=-1)
#Legend
legend_elements = []
for i in range(len(normal)):
    legend_elements.append(Line2D([0],[0],ls='-',color=colors_mp[i],label=list_labels[i],linewidth=1))
legend_elements.append(Line2D([0],[0],ls='--',color=colors_gr[0],
    label='devices',linewidth=1))
ax_big.legend(handles=legend_elements,loc='lower right')
if 0:   #FGT injection
    for i in lis:  #FGT graphs
        plt.plot(data2[i][:,0],data2[i][:,1]/np.max(data2[i][:,1]),ls='--',label='FGT dev '+str(i+2))
#Shaded areas
shadeAA = 'goldenrod'
shadeM = 'chartreuse'
ax_big.fill_betweenx(np.linspace(0,1,100),0.1,0.3,facecolor=shadeAA, alpha = 0.3)
ax_big.fill_betweenx(np.linspace(0,1,100),0.6,0.8,facecolor=shadeM, alpha = 0.3)
#Axis
ax_big.set_xlabel(r'$B[T]$',size=20)
ax_big.set_ylabel(r'$\delta G$',size=20,labelpad=-10,rotation=90)
ax_big2 = ax_big.twinx()
if squared:
    #ax_big2.set_ylabel(r'$k\left(\frac{M_z(B)-M_z(B=0)}{M_z(B=2T)-M_z(B=0)}\right)^2$',size=20,rotation=90)
    ax_big2.set_ylabel(r'$(\Delta M_z)^2$',size=20,rotation=90)
else:
    #ax_big2.set_ylabel(r'$\frac{M_z(B)-M_z(B=0)}{M_z(B=2T)-M_z(B=0)}$',size=20)
    ax_big2.set_ylabel(r'$\Delta M_z$',size=20,rotation=90)
ax_big.spines['left'].set_visible(False)
ax_big2.spines['left'].set_linestyle((0,(5,5)))
ax_big2.spines['left'].set_color('cornflowerblue')
ax_big.spines['right'].set_visible(False)
ax_big2.spines['right'].set_color('firebrick')
#ax_big.set_yticks([])
#Limits and font size
ax_big.set_xlim(0,1.2)
ax_big.set_ylim(0,1)
ax_big2.set_ylim(0,1)
ax_big.tick_params(axis='both',which='major',labelsize=20)
ax_big2.tick_params(axis='both',which='major',labelsize=20)
#Text
ax_big.text(0.01,0.7,r'$(I)$',size=20)
ax_big.text(0.35,0.7,r'$(II)$',size=20)
ax_big.text(1.0,0.7,r'$(III)$',size=20)
if 0:   #Arrows to axis
    arrow1 = patches.FancyArrow(0.3,0.4,-0.3,0.1,
            ls='dashed',
            width=0.0005,
            length_includes_head=True,
            head_width=0.02,
            head_length=0.03,
            edgecolor='aqua',
            facecolor='aqua',
            )
    ax_big.add_patch(arrow1)
    arrow2 = patches.FancyArrow(0.8,0.6,0.4,-0.1,
    #        ls='dashed',
            width=0.0005,
            length_includes_head=True,
            head_width=0.02,
            head_length=0.03,
            edgecolor='firebrick',
            facecolor='firebrick',
            )
    ax_big.add_patch(arrow2)


########################################################################
########################################################################
if 1:
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

    #Colored patches to indicate shaded area
    arr_style = patches.ArrowStyle("Fancy", head_length=7, head_width=7, tail_width=0.1)
    XX = 0.25
    lll = 0.1+(0.275-XX)*2
    YY = 0.94
    ax = fig.add_subplot(gs[0,:3])
    ax.set_axis_off()
    rectangle = patches.Rectangle((0.275,YY+0.03),0.1,-0.95,edgecolor='none',facecolor=shadeAA,alpha=0.3)
    ax.add_patch(rectangle)
    rectangle = patches.Rectangle((0.275+0.35,YY+0.03),0.1,-0.95,edgecolor='none',facecolor=shadeM,alpha=0.3)
#    rectangle = patches.Rectangle((0.275+0.35,YY-0.05),0.1,0.1,edgecolor='none',facecolor=shadeM,alpha=0.3)
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
#    ax.text(0.128,0.05,r'$(I)$',size=20)
#    ax.text(0.472,0.05,r'$(II)$',size=20)
#    ax.text(0.818,0.05,r'$(III)$',size=20)

    #Single Hexagons
    for i in range(3):
        ax = fig.add_subplot(gs[0,i])
        ax.set_axis_off()
        a1 = np.array([1,0])*2
        a2 = np.array([-1/2,np.sqrt(3)/2])*2
        if 1:       #Moire lattice
            rows = 20
            cols = 20
            radius1 = 0.08      #lattice constant
            radius2 = radius1*(radius1+1)
            linewidth = 0.05
            for i_ in range(-rows,rows):
                for j_ in range(-cols,cols):
                    x = j_ * radius1 * np.sqrt(3) - i_ * radius1 / 2 * np.sqrt(3)
                    y = i_ * radius1 * 1.5
                    hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=radius1, orientation=0, edgecolor='k', facecolor='none', zorder=3, linewidth=linewidth)
                    ax.add_patch(hexagon)
                    x = j_ * radius2 * np.sqrt(3) - i_ * radius2 / 2 * np.sqrt(3)
                    y = i_ * radius2 * 1.5
                    hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=radius2, orientation=0, edgecolor='k', facecolor='none', zorder=3, linewidth=linewidth)
                    ax.add_patch(hexagon)
        #Interlayer interaction contour at 0
        X,Y = np.meshgrid(np.linspace(-1,1,400),np.linspace(-1,1,400))
        A1 = X*a1[0] + Y*a2[0]
        A2 = X*a1[1] + Y*a2[1]
        Phi = np.load('Data/CrBr3_interlayer_rescaled.npy')
        big_Phi = np.zeros((400,400))
        for i_ in range(2):
            for j_ in range(2):
                big_Phi[i_*200:(i_+1)*200,j_*200:(j_+1)*200] = Phi
        con = ax.contour(A1,A2,big_Phi.T,levels=[0,],colors=('r',),linestyles=('-',),linewidths=(0.5,))
        if 1:
            #Phase values
            X,Y = np.meshgrid(np.linspace(-1,1,600),np.linspace(-1,1,600))
            A1 = X*a1[0] + Y*a2[0]
            A2 = X*a1[1] + Y*a2[1]
            list_M = [0,13,60]
            big_sol = np.zeros((600,600))
            for i_ in range(2):
                for j_ in range(2):
                    big_sol[i_*300:(i_+1)*300,j_*300:(j_+1)*300] = sol[list_M[i]][1]
            if i == 0:
                max_Phi = np.max(big_sol)
                min_Phi = np.min(big_sol)
            cof = ax.contourf(A1,A2,big_sol,levels=np.linspace(0,np.pi,100),vmin=0,vmax=np.pi,cmap='viridis_r',extend='both',alpha=1)
        #UC corners
        ax.set_aspect(1.)
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

        if 1:
            all_t = [
                [(1,0),(0,0),(1,0),(1,0)],
                [(1,0),(0,0),(0,0),(2,0)],
                [(3,2),(0,0),(0,0),(5,4)],
                ]
            #Spins
            width = 0.33
            height = 0.68
            xs = [-0.6,0.3,-0.15,-0.15]
            ys = [-0.3,-0.3,0.2,-0.8]
            rect_edgecolor = 'k'
            rect_facecolor = 'none'
            ddd = height/20
            arr_len = height/2-height/5
            ang = np.pi/5
            ang2 = np.pi/3
            i_arr = [       #initial position of arrows
                    (width/2,ddd),
                    (width/2,ddd+arr_len+0.1),
                    (width/2+arr_len/2*np.cos(ang),height/4-arr_len/2*np.sin(ang)),
                    (width/2-arr_len/2*np.cos(ang),height/4-arr_len/2*np.sin(ang)),
                    (width/2+arr_len/2*np.cos(ang2),height/4-arr_len/2*np.sin(ang2)),
                    (width/2-arr_len/2*np.cos(ang2),height/4-arr_len/2*np.sin(ang2)),
                    ]
            disps = [(0,arr_len),(0,-arr_len),
                    (-arr_len*np.cos(ang),arr_len*np.sin(ang)),(arr_len*np.cos(ang),arr_len*np.sin(ang)),
                    (-arr_len*np.cos(ang2),arr_len*np.sin(ang2)),(arr_len*np.cos(ang2),arr_len*np.sin(ang2)),
                    ]
            tts = all_t[i]
            arr_color = ['orangered','b','g','g','limegreen','limegreen']
            curved_arrow = patches.FancyArrowPatch((xs[0]+width/2,ys[0]+0.02),(xs[0]+0.2,ys[0]-0.45),
                    connectionstyle='arc3,rad=.2',
                    color='k',
                    arrowstyle='Simple, tail_width=0.5, head_width=4,head_length=8',
                    zorder=5,
                    )
            ax.add_patch(curved_arrow)
            curved_arrow = patches.FancyArrowPatch((xs[1]+width/2,ys[1]+height-0.02),(xs[1],ys[1]+height+0.2),
                    connectionstyle='arc3,rad=.5',
                    color='k',
                    arrowstyle='Simple, tail_width=0.5, head_width=4,head_length=8',
                    zorder=5,
                    )
            ax.add_patch(curved_arrow)
            curved_arrow = patches.FancyArrowPatch((xs[2]+0.01,ys[2]+0.02),(0,0),
                    connectionstyle='arc3,rad=.3',
                    color='k',
                    arrowstyle='Simple, tail_width=0.5, head_width=4,head_length=8',
                    zorder=5,
                    )
            ax.add_patch(curved_arrow)
            curved_arrow = patches.FancyArrowPatch((xs[3]+width-0.01,ys[3]+height/2),(0.5,-0.65),
                    connectionstyle='arc3,rad=-.3',
                    color='k',
                    arrowstyle='Simple, tail_width=0.5, head_width=4,head_length=8',
                    zorder=5,
                    )
            ax.add_patch(curved_arrow)
            for sss in range(4):
                X1 = xs[sss]
                Y1 = ys[sss]
                rectangle = patches.Rectangle((X1,Y1),width,height,edgecolor=rect_edgecolor,facecolor=rect_facecolor,linewidth=1.5)
                ax.add_patch(rectangle)
                for ik in range(2):
                    din = i_arr[tts[sss][ik]]
                    xa = X1+din[0]
                    ya = Y1+din[1]+height/2*ik
                    disp = disps[tts[sss][ik]]
#                    dot = Line2D([(2*xa+disp[0])/2],[(2*ya+disp[1])/2],marker='o',color=arr_color[tts[sss][ik]],markerfacecolor=arr_color[tts[sss][ik]],markersize=8,linewidth=0,zorder=5)
                    dot = patches.Circle(((2*xa+disp[0])/2,(2*ya+disp[1])/2),radius=0.035,
                            linewidth=0.1,
                            edgecolor='k',#arr_color[tts[sss][ik]],
                            facecolor=arr_color[tts[sss][ik]],
                            alpha=1,
                            zorder=5,
                            )
                    arrow = patches.FancyArrow(xa,ya,disp[0],disp[1],
                            linewidth=0.1,
                            width=0.022,
                            length_includes_head=False,
                            head_width=0.11,
                            head_length=0.09,
                            edgecolor='k',#arr_color[tts[sss][ik]],
                            facecolor=arr_color[tts[sss][ik]],
                            zorder=4,
                            alpha=1
                            )
                    ax.add_patch(arrow)
                    ax.add_patch(dot)
    #Colorbar
    ax = fig.add_subplot(gs[:2,0:3])
    ax.set_axis_off()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    cbar = fig.colorbar(cof,ax=ax,orientation='horizontal',location='bottom',ticks=[min_Phi,max_Phi],
            anchor = (0.5,.5),
            aspect = 30,
            )
    cbar.ax.set_xticklabels(['$1$','$-1$'],size=20)
    ax.text(0.5,-0.5,r'$M_z$',fontsize=30,zorder=5)

fig.tight_layout()
plt.subplots_adjust(wspace=0.3)

if 0:
    plt.savefig('/home/dario/Desktop/figs_crbr/picture6.pdf')
    print("Saved")
    exit()
else:
    plt.show()
