import numpy as np
from pathlib import Path
import functions as fs
import h5py
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

shadeAA = 'goldenrod'
shadeM = 'chartreuse'

def plot_hexagon(ax,i,sol,**kwargs):
    if 'moire' in kwargs:
        plot_moire = kwargs['moire']
    else:
        plot_moire = False
    if 'background' in kwargs:
        plot_background = kwargs['background']
    else:
        plot_background = False
    if 'panels' in kwargs:
        plot_panels = kwargs['panels']
    else:
        plot_panels = False
    ax.set_axis_off()
    a1 = np.array([1,0])*2
    a2 = np.array([-1/2,np.sqrt(3)/2])*2
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
    if plot_moire:       #Moire lattice
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
    if plot_background:
        #Phase values
        X,Y = np.meshgrid(np.linspace(-1,1,600),np.linspace(-1,1,600))
        A1 = X*a1[0] + Y*a2[0]
        A2 = X*a1[1] + Y*a2[1]
        list_M = [0,13,60]
        big_sol = np.zeros((600,600))
        for i_ in range(2):
            for j_ in range(2):
                big_sol[i_*300:(i_+1)*300,j_*300:(j_+1)*300] = sol[list_M[i]][1]
        max_Phi = np.max(big_sol)
        min_Phi = np.min(big_sol)
        cof = ax.contourf(A1,A2,big_sol,levels=np.linspace(0,np.pi,100),vmin=0,vmax=np.pi,cmap='viridis_r',extend='both',alpha=1)
    if plot_panels:
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
    return cof, min_Phi,max_Phi

def colorbar(fig,ax,cof,min_Phi,max_Phi):
    ax.set_axis_off()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    cbar = fig.colorbar(cof,ax=ax,orientation='horizontal',location='bottom',ticks=[min_Phi,max_Phi],
            anchor = (0.5,.5),
            aspect = 30,
            )
    cbar.ax.set_xticklabels(['$1$','$-1$'],size=20)
    ax.text(0.5,-0.5,r'$M_z$',fontsize=30,zorder=5)

def theory_exp_plot(ax_big,data_theory,mps,list_par,data_exp,ylabel):
    colors_mp = ['red','brown','orange','maroon','darkorange','orangered','firebrick','maroon']
    label_dic = {'b':'Biaxial','u':'Uniaxial'}
    list_labels = []
    for i in range(len(data_theory)):
        X = mps[i][:,0]
        pp = list_par[i]
        list_labels.append(label_dic[pp[0]]+': '+r'$\epsilon=$'+"{:.0f}".format(pp[2]*100)+'\%, '+r'$\rho=$'+"{:.1f}".format(pp[3])+' meV, '+r'$d=$'+"{:.2f}".format(pp[4])+' meV')
        ax_big.plot(X,data_theory[i],color=colors_mp[i])
        ax_big.scatter(X[pp[-2]],data_theory[i][pp[-2]],marker='*',color=colors_mp[i])
        ax_big.scatter(X[pp[-1]],data_theory[i][pp[-1]],marker='*',color=colors_mp[i])
    lis = np.arange(4)
    colors_gr = ['paleturquoise','turquoise','c','darkcyan']
    for i in lis:
        x = data_exp[i][:,0]
        ax_big.plot(x,data_exp[i][:,1]/data_exp[i][-1,1],ls='--',
                color=colors_gr[i],
                zorder=-1)
    #Legend
    legend_elements = []
    for i in range(len(data_theory)):
        legend_elements.append(Line2D([0],[0],ls='-',color=colors_mp[i],label=list_labels[i],linewidth=1))
    legend_elements.append(Line2D([0],[0],ls='--',color=colors_gr[0],
        label='devices',linewidth=1))
    ax_big.legend(handles=legend_elements,loc='lower right')
    #Shaded areas
    ax_big.fill_betweenx(np.linspace(0,1,100),0.1,0.3,facecolor=shadeAA, alpha = 0.3)
    ax_big.fill_betweenx(np.linspace(0,1,100),0.6,0.8,facecolor=shadeM, alpha = 0.3)
    #Axis
    ax_big.set_xlabel(r'$B[T]$',size=20)
    ax_big.set_ylabel(r'$\delta G$',size=20,labelpad=-10,rotation=90)
    ax_big2 = ax_big.twinx()
    ax_big2.set_ylabel(ylabel,size=20,rotation=90)
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

def import_sol(sol_par,sol_fn):
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
    return sol

def extract_exp_data(save1,save2):
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

    return data1, data2


def import_theory_data(MP1_par,MP1_fn):
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
    np.save(MP1_fn,np.array(MP1))
    return np.array(MP1)

