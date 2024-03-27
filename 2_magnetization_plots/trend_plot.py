import numpy as np
import sys
import functions as fs
import MP_features as ft
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

if not len(sys.argv)==4:
    print('Inputs needed: u/b/B - rae - M/AA')
    exit()


rhos = fs.rhos
anis = fs.anis
epss = fs.epss
data = {'r':[rhos,r'$\rho$',["{:.1f}".format(rhos[i]) for i in range(len(rhos))],1],
        'a':[anis,r'$d$',["{:.3f}".format(anis[i]) for i in range(len(anis))],2],
        'e':[epss,r'$\epsilon$',["{:.1f}".format(epss[i]*100) for i in range(len(epss))],0]
        }

s1,s2,s3 = [*sys.argv[2]]       #rae
TT = sys.argv[3]                #M or AA

unit = ' meV' if s1 == 'r' else ''
a1,l1,v1,ind1 = data[s1]
a2,l2,v2,ind2 = data[s2]
a3,l3,v3,ind3 = data[s3]

#Features
M_bi = ft.M_bi
AA_bi = ft.AA_bi
AA_bi_flip = ft.AA_bi_flip
AA_bi_flop = ft.AA_bi_flop
M_uni = ft.M_uni[:,0,:,:]        #just tr=0 for now
AA_uni = ft.AA_uni[:,0,:,:]
AA_uni_flip = ft.AA_uni_flip[:,0,:,:]
AA_uni_flop = ft.AA_uni_flop[:,0,:,:]
#
M_bi_ordered = np.swapaxes(M_bi,0,ind1)
AA_bi_ordered = np.swapaxes(AA_bi,0,ind1)
AAflip_bi_ordered = np.swapaxes(AA_bi_flip,0,ind1)
AAflop_bi_ordered = np.swapaxes(AA_bi_flop,0,ind1)
if not ind3 == 0:
    M_bi_ordered = np.swapaxes(M_bi_ordered,1,ind3)
    AAflip_bi_ordered = np.swapaxes(AAflip_bi_ordered,1,ind3)
    AAflop_bi_ordered = np.swapaxes(AAflop_bi_ordered,1,ind3)
    AA_bi_ordered = np.swapaxes(AA_bi_ordered,1,ind3)
M_uni_ordered = np.swapaxes(M_uni,0,ind1)
AA_uni_ordered = np.swapaxes(AA_uni,0,ind1)
AAflip_uni_ordered = np.swapaxes(AA_uni_flip,0,ind1)
AAflop_uni_ordered = np.swapaxes(AA_uni_flop,0,ind1)
if not ind3 == 0:
    M_uni_ordered = np.swapaxes(M_uni_ordered,1,ind3)
    AAflip_uni_ordered = np.swapaxes(AAflip_uni_ordered,1,ind3)
    AAflop_uni_ordered = np.swapaxes(AAflop_uni_ordered,1,ind3)
    AA_uni_ordered = np.swapaxes(AA_uni_ordered,1,ind3)
datas = {'M':[M_bi_ordered,M_uni_ordered],'AA':[AA_bi_ordered,AA_uni_ordered]}
datas_flip = [AAflip_bi_ordered,AAflip_uni_ordered]
datas_flop = [AAflop_bi_ordered,AAflop_uni_ordered]
#Use biaxial data for axis limits in any cas -> can be corrected if needed
plotted = M_bi_ordered if TT=='M' else AA_bi_ordered
ylims = [np.min(plotted[~np.isnan(plotted)]),np.max(plotted[~np.isnan(plotted)])]
ylims[0] -= (ylims[1]-ylims[0])/10
ylims[1] += (ylims[1]-ylims[0])/10
if TT=='AA':
    ylims[1]=0.35
shade = 'chartreuse' if TT=='M' else 'khaki'
shadelim = (0.6,0.95) if TT=='M' else (0.1,0.3)
mark = {'M':'o','AA':''}
mark2 = ['.','*']
mark3 = [['.','*'],['^','X']]

############################################################################################################
############################################################################################################
############################################################################################################
#PLOT
n_cols = 6 if s1 in ['r','a'] else 3
fig,axs = plt.subplots(nrows=2,ncols=n_cols)
fig.set_size_inches(16,9)
gs = axs[0,0].get_gridspec()
for i in range(n_cols):
    for ax in axs[:2,i]:
        ax.remove()
colors = ['r','b','g','y','m','c']
colors_2 = ['firebrick','darkblue','darkgreen','olive','darkorchid','darkcyan']
markersize = 50
for i1 in range(len(a1)):
    i1x = i1//3
    i1y1 = i1%3 if s1=='e' else (i1%3)*2+i1x
    i1y2 = i1%3+1 if s1=='e' else (i1%3)*2+2+i1x
    ax = fig.add_subplot(gs[i1x,i1y1:i1y2])
    for i3 in range(len(a3)):
        if sys.argv[1] == 'b':
            ax.plot(a2,datas[TT][0][i1,i3,:],marker=mark[TT],ls='--',label=l3+'='+v3[i3],color=colors[i3],linewidth=0.5)
        elif sys.argv[1] == 'u':
            ax.plot(a2,datas[TT][1][i1,i3,:],marker=mark[TT],ls='--',label=l3+'='+v3[i3],color=colors[i3],linewidth=0.5)
        elif sys.argv[1] == 'B':
            for step in range(2):
                ax.plot(a2,datas[TT][step][i1,i3,:],marker=mark2[step],ls='--',label=l3+'='+v3[i3],color=colors[i3],linewidth=0.5)
        if TT == 'AA':
            if sys.argv[1] == 'b':
                ax.scatter(a2,datas_flip[0][i1,i3,:],marker='*',ls='--',color=colors[i3],linewidth=0.5,s=markersize)
                ax.scatter(a2,datas_flop[0][i1,i3,:],marker='^',ls='--',color=colors[i3],linewidth=0.5,s=markersize)
            elif sys.argv[1] == 'u':
                ax.scatter(a2,datas_flip[1][i1,i3,:],marker='*',ls='--',color=colors[i3],linewidth=0.5,s=markersize)
                ax.scatter(a2,datas_flop[1][i1,i3,:],marker='^',ls='--',color=colors[i3],linewidth=0.5,s=markersize)
            elif sys.argv[1] == 'B':
                for step in range(2):
                    ax.scatter(a2,datas_flip[step][i1,i3,:],marker=mark3[step][0],ls='--',color=colors[i3],linewidth=0.5,s=markersize)
                    ax.scatter(a2,datas_flop[step][i1,i3,:],marker=mark3[step][1],ls='--',color=colors[i3],linewidth=0.5,s=markersize)
    ax.set_title(l1+'='+v1[i1]+unit,size=20)
    if i1 in [3,4,5]:
        ax.set_xlabel(l2,size=20)
    else:
        ax.set_xticklabels([])
    if i1 in [0,3]:
        ax.set_ylabel(r'$h_\bot^M[T]$',size=20)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis='both',which='major',labelsize=20)
    ax.set_ylim(*ylims)
    pos = ax.get_xlim()
    ax.set_xlim(pos)
    ax.fill_between(np.linspace(pos[0],pos[1],100),shadelim[0],shadelim[1],facecolor=shade, alpha = 0.3)

if not s1=='e':
    from matplotlib.lines import Line2D
    #Add for AA the legend with flip/flop markers
    legend_elements = []
    for i in range(6):
        legend_elements.append(Line2D([0],[0],ls='--',color=colors[i],label=r'$\epsilon=$'+v3[i]+'\%',linewidth=1))
    if TT == 'AA':
        legend_elements += [ Line2D([0],[0],marker='.',color='k',label='biaxial flip',markerfacecolor='k',markersize=8,linewidth=0),
                            Line2D([0],[0],marker='*',color='k',label='biaxial flop',markerfacecolor='k',markersize=8,linewidth=0),
                            Line2D([0],[0],marker='^',color='k',label='uniaxial flip',markerfacecolor='k',markersize=8,linewidth=0),
                            Line2D([0],[0],marker='X',color='k',label='uniaxial flop',markerfacecolor='k',markersize=8,linewidth=0),
                        ]
    if TT == 'M':
        legend_elements += [ Line2D([0],[0],marker='.',color='k',label='biaxial',markerfacecolor='k',markersize=8,linewidth=0),
                            Line2D([0],[0],marker='*',color='k',label='uniaxial',markerfacecolor='k',markersize=8,linewidth=0),
                        ]
    ax = fig.add_subplot(gs[1,5])
    ax.set_axis_off()
    ax.legend(handles=legend_elements,fontsize=20)


fig.tight_layout()
if 0:
    plt.savefig('/home/dario/Desktop/figs_crbr/trend_'+sys.argv[1]+'_'+TT+'_'+sys.argv[2]+'.pdf')
    print("Saved")
    exit()
else:
    plt.show()

