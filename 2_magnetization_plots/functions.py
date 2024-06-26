import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams.update({"text.usetex": True,})
s_ = 20
import random
import os
import h5py
from time import time
import matplotlib as mpl
def_cmap = mpl.colormaps['RdBu']

#Physical parameters
rho_phys = {'DFT':1.4,'exp':1.7} #     (meV)      
d_phys = {'DFT':0.0709,'exp':0.09} #     (meV)       0.0709
#
gammas = {  'MPs':np.linspace(0,2,100,endpoint=False), 
            'AA':np.linspace(0,0.8,100,endpoint=False),
            'M':np.linspace(0,0.8,100,endpoint=False),
            }

conversion_factor = 0.607 
Spin = 3/2
#
rhos = [0.1,1.4,5,10,100]
anis = [0.01,0.03,0.0709,0.11,0.15]

epss = [0.05,0.04,0.03,0.02,0.01,0.005]
translations = [0,0.1,0.2,1/3,0.4,0.5]
nis = [0,0.1,1/3,2/3]

#
thetas = np.pi/180*0
#
NNNN = 21
lr_list = np.logspace(-5,1,num=NNNN)

nn_n = 2
cutoff = 1e-8

list_ind = {'PDb':
            np.arange(49),
            
            'PDu':
            np.arange(49),
            
            'CO':   np.arange(100),
            }

#Triangular lattice
a1 = np.array([1,0])
a2 = np.array([-1/2,np.sqrt(3)/2])
d_vec = np.array([0,1/np.sqrt(3)])  #vector connecting the two sublattices
b1 = np.array([1,1/np.sqrt(3)])*2*np.pi
b2 = np.array([0,2/np.sqrt(3)])*2*np.pi

def const_in_pt(fA,fB,gx,gy):
    """Computes the initial point for the minimization. The possibilities are for now
    either twisted-s -> ans=0, or constant -> ans=1

    Parameters
    ----------
    Phi : np.ndarray
        Interlayer coupling of size (grid,grid)
    pars : 3-tuple
        Parameters alpha, beta and gamma.
    fs : float
        Random number between 0 and 1 to set inisial condition for symmetric phase.
    fa : float
        Random number between 0 and 1 to set inisial condition for a-symmetric phase.
    ans : int
        Index to chose the ansatz of the initial point.

    Returns
    -------
    List
        Symmetric and antisymmetric phases at each position (grid,grid) of the Moirè unit cell.
    """
    phi_A = np.ones((gx,gy))*fA
    phi_B = np.ones((gx,gy))*fB
    return np.array([phi_A, phi_B])

def compute_solution(args_m):
    """Computes the magnetization pattern by performing a gradient descent from random 
    initial points.

    Parameters
    ----------
    Phi: np.ndarray
        Interlayer coupling of size (grid,grid)
    pars: 3-tuple
        Parameters alpha, beta and gamma.
    args_minimization : dic
        'rand_m' -> int, number of random initial seeds,
        'maxiter' -> int, max number of update evaluations,
        'cluster_name' -> string, name of machine,
        'disp' -> bool, diplay messages.

    Returns
    -------
    np.ndarray

        Symmetric and antisymmetric phases at each position (grid,grid) of the Moirè unit cell.
    """
    Phi,A_M = args_m['args_moire']
    a1_m, a2_m = A_M
    M_transf = get_M_transf(a1_m,a2_m)
    gx,gy = args_m['grid']
    disp = args_m['disp']
    solution_fn = get_sol_fn(args_m['moire_pars'],args_m['grid'],args_m['args_phys'],args_m['machine'])
    #Variables for storing best solution
    min_E = 1e10
    result = np.ones((2,gx,gy))*20
    initial_index = 0
    print("gamma: ",args_m['args_phys'])
    for ind_in_pt in range(initial_index,args_m['n_initial_pts']):  #############
        t0 = time()
        if 1 and disp:
            print("Starting minimization step ",str(ind_in_pt))
        #Initial condition
        inddd = list_ind[args_m['type_comp']][ind_in_pt]
        f1 = ((inddd//7)*180/7+7)/180*np.pi
        f2 = ((inddd%7)*180/7+7)/180*np.pi
        phi = const_in_pt(f1,f2,gx,gy)
        #First energy evaluation
        E = [compute_energy(phi,Phi,args_m['args_phys'],A_M,M_transf), ]
        #Initialize learning rate and minimization loop
        step = 1        #initial step
        keep_going = True
        while keep_going:
            #Energy gradients
            dH = grad_H(phi,Phi,args_m['args_phys'],A_M,M_transf)
            if 0 and disp:
                plot_phis(dH,A_M,'grad')
            #Compute energy in all points of LR
            list_E = []
            list_phi = []
            for lr_i in range(NNNN):
                LR_ = lr_list[lr_i]
                phi_new = np.copy(phi-LR_*dH)
                temp_E = compute_energy(phi_new,Phi,args_m['args_phys'],A_M,M_transf)
                list_phi.append(np.copy(phi_new))
                list_E.append(np.array([LR_,temp_E]))
            list_E = np.array(list_E)
            #Check the minimum of energies wrt LR
            amin = np.argmin(list_E[:,1])
            if list_E[amin,1] < E[0]:
                E.insert(0,list_E[amin,1])
                phi = np.copy(list_phi[amin])
                if 0 and disp:
                    print("step: ",step," with E:","{:.15f}".format(E[0]))
            else:
                print(ind_in_pt," NOT CONVERGED")
                keep_going = False
            #Check if energy converged to a constant value
            if check_energies(E):
                if E[0]<min_E:
                    min_E = E[0]
                    result = np.copy(phi)
                    print("\tindex ",ind_in_pt," is new solution with energy ","{:.8f}".format(min_E))
                    if not args_m['machine']=='loc':
                        np.save(solution_fn,result)
                    if 1 and disp:
                        print("mag: ",compute_magnetization(phi))
                   #     input()
                else:
                    print(ind_in_pt," at higher energy: ","{:.8f}".format(E[0]))
                keep_going = False
            if step > args_m['maxiter']:
                print(ind_in_pt," reached maxiter")
                keep_going = False
            step += 1
    if (result == np.ones((2,gx,gy))*20).all():
        print("Not a single converged solution, they all reached max number of iterations or too low LR")
        exit()
    return result

def compute_energy(phi,Phi,phys_args,A_M,M_transf):
    """Computes the energy of the system.

    Parameters
    ----------
    phi: 2-tuple
        Symmetric and Anti-Symmetric phases.
    Phi : np.ndarray
        Interlayer coupling of size (grid,grid)
    pars : 3-tuple
        Parameters alpha, beta and gamma.

    Returns
    -------
    float
        Energy density summed over all sites.
    """
    gamma,rho,anisotropy = phys_args
    det, n1x, n2x, n1y, n2y = M_transf
    a1_m, a2_m = A_M
    gx,gy = phi[0].shape
    xx = np.linspace(0,np.linalg.norm(a1_m),gx,endpoint=False)
    yy = np.linspace(0,np.linalg.norm(a2_m),gy,endpoint=False)
    dx = xx[1]-xx[0]
    dy = yy[1]-yy[0]
    grad_2 = []
    for i in range(2):
        d_phi1 = derivative(phi[i],dx,0)
        d_phi2 = derivative(phi[i],dy,1)
        #
        grad_2.append( (n1x*d_phi1+n2x*d_phi2)**2+(n1y*d_phi1+n2y*d_phi2)**2 )
    energy = rho/2*grad_2[0]+rho/2*grad_2[1] - anisotropy*(np.cos(phi[0])**2+np.cos(phi[1])**2) - Phi*np.cos(phi[0]-phi[1]) - conversion_factor/Spin*gamma*(np.cos(phi[0])+np.cos(phi[1]))
    return energy.sum()/gx/gy

def grad_H(phi,Phi,phys_args,A_M,M_transf):
    """Computes evolution step dH/d phi.

    Parameters
    ----------
    phi: 2-tuple
        Symmetric and Anti-Symmetric phases.
    tt : char
        Determines which functional derivative to compute (wrt phi_s or phi_a).
    Phi : np.ndarray
        Interlayer coupling of size (grid,grid)
    pars : 3-tuple
        Parameters alpha, beta and gamma.
    d2_phi : ndarray
        Second derivative of the phase (symm or anti-symm) to compute.

    Returns
    -------
    np.ndarray
        Gradient of Hamiltonian on the (grid,grid) space.
    """
    gamma,rho,anisotropy = phys_args
    det, n1x, n2x, n1y, n2y = M_transf
    a1_m, a2_m = A_M
    gx,gy = phi[0].shape
    xx = np.linspace(0,np.linalg.norm(a1_m),gx,endpoint=False)
    yy = np.linspace(0,np.linalg.norm(a2_m),gy,endpoint=False)
    dx = xx[1]-xx[0]
    dy = yy[1]-yy[0]
    res = np.zeros((2,gx,gy))
    for i in range(2):
        tt_phi = phi[i]
        yy_phi = phi[(i+1)%2]
        d_phi11 = derivative2(tt_phi,dx,0)
        d_phi22 = derivative2(tt_phi,dy,1)
        d_phi12 = derivative(derivative(tt_phi,dx,0),dy,1)
        lapl = (n1x**2+n1y**2)*d_phi11 + 2*(n1x*n2x+n1y*n2y)*d_phi12 + (n2x**2+n2y**2)*d_phi22
        res[i] = -rho*lapl + anisotropy*np.sin(2*tt_phi) + Phi*np.sin(tt_phi-yy_phi) + conversion_factor/Spin*gamma*np.sin(tt_phi)
    return res

def derivative(phi,dd,ax):
    return (1/280*np.roll(phi,4,axis=ax)-4/105*np.roll(phi,3,axis=ax)+1/5*np.roll(phi,2,axis=ax)-4/5*np.roll(phi,1,axis=ax)+4/5*np.roll(phi,-1,axis=ax)-1/5*np.roll(phi,-2,axis=ax)+4/105*np.roll(phi,-3,axis=ax)-1/280*np.roll(phi,-4,axis=ax))/dd

def derivative2(phi,dd,ax):
    return (-1/560*np.roll(phi,4,axis=ax)+8/315*np.roll(phi,3,axis=ax)-1/5*np.roll(phi,2,axis=ax)+8/5*np.roll(phi,1,axis=ax)-205/72*phi+8/5*np.roll(phi,-1,axis=ax)-1/5*np.roll(phi,-2,axis=ax)+8/315*np.roll(phi,-3,axis=ax)-1/560*np.roll(phi,-4,axis=ax))/dd**2

def get_M_transf(a1_m,a2_m):
    det = a1_m[0]*a2_m[1]-a2_m[0]*a1_m[1]
    n1x = a2_m[1]/det
    n2x = -a1_m[1]/det
    n1y = -a2_m[0]/det
    n2y = a1_m[0]/det
    return (det,n1x,n2x,n1y,n2y)

def check_energies(list_E):
    """ Checks wether the last nn energies in the list_E are within lim distance to each other.

    Parameters
    ----------
    list_E: list
        List of energies to check.

    Returns
    -------
    bool
        True if last nn energies are within lim distance b/w each other, False otherwise.
    
    """
    if len(list_E) <= nn_n:
        return False
    for i in range(nn_n):
        if abs(list_E[i]-list_E[i+1]) > cutoff:
            return False
    return True

def plot_magnetization(phi,Phi,A_M,gamma,**kwargs):
    """Plots the magnetization values in the Moirè unit cell, with a background given by the
    interlayer potential. The two images correspond to the 2 layers. Magnetization is in x-z
    plane while the layers are in x-y plane.

    Parameters
    ----------
    phi: 2-tuple
        Symmetric and Anti-Symmetric phases.
    Phi : np.ndarray
        Interlayer coupling of size (grid,grid).
    pars : 3-tuple
        Parameters alpha, beta and gamma.
    save : bool (optional, default=False)
        Wether to save or not the plot.
    tt : string (optional)
        Name of the figure (just needed if save=True).

    """
    gx,gy = phi[0].shape
    a1_m,a2_m = A_M
    a12_m = a1_m+a2_m
    #Single layer phases
    phi_1 = phi[0]
    phi_2 = phi[1]
    #Background -> interlayer coupling
    nn = 5
    big_Phi = extend(Phi,nn)
    long_X = np.linspace(-nn//2,nn//2,nn*gx,endpoint=False)
    long_Y = np.linspace(-nn//2,nn//2,nn*gy,endpoint=False)
    X,Y = np.meshgrid(long_X,long_Y)
    A1 = X*A_M[0][0] + Y*A_M[1][0]
    A2 = X*A_M[0][1] + Y*A_M[1][1]
    #BZ lines parameters
    x_i,args_i = get_BZ_args(a1_m,a2_m,a12_m)
    mi,qi = get_l_args(args_i)
    #Arrows parameters
    l = np.linalg.norm(a1_m)/40 if np.linalg.norm(a1_m)>np.linalg.norm(a2_m) else np.linalg.norm(a2_m)/40#0.02       #length of arrow
    hw = l/2#0.01       #arrow head width
    hl = l/2#0.01       #arrow head length
    facx = gx//30#12     #plot 1 spin every "fac" of grid
    facy = gy//30#12 #if gy>=10 else 1     #plot 1 spin every "fac" of grid
    phi_ = [phi_1,phi_2]
    #Figure
    #fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(10,5))
    fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(20,10))
    for ind,ax in enumerate([ax1,ax2]):
        ax.axis('off')
        ax.set_aspect(1.)
        ax.contour(A1,A2,big_Phi.T,levels=[0,],colors=('k',),linestyles=('-',),linewidths=(0.5,))
        surf = ax.contourf(A1,A2,big_Phi.T,levels=100,cmap=def_cmap)
#        plt.colorbar(surf)
        #Box unit cell
        if 1:
            #ax.arrow(0,0,a1_m[0],a1_m[1])
            #ax.arrow(0,0,a2_m[0],a2_m[1])
            for i in range(3):
                for s in [-1,1]:
                    li = line(np.linspace(-x_i[i],x_i[i],100),args_i[i],s)
                    ax.plot(li[:,0],li[:,1],'k',lw=0.7,ls='--',dashes=(7,20))
        #plot small arrows
        for i in range(gx//facx):
            for j in range(gy//facy if facy>0 else 1):
                x_c = (i*facx)/gx
                y_c = (j*facy)/gy
                x = x_c*A_M[0][0] + y_c*A_M[1][0]
                y = x_c*A_M[0][1] + y_c*A_M[1][1]
                x,y = inside_UC(x,y,mi,qi,a1_m,a2_m,a12_m)
                phi_fin = phi_[ind][i*facx,j*facy]
                aa = np.copy(phi_fin)
                if aa < 0:
                    aa += 2*np.pi
                if aa > np.pi:
                    aa = 2*np.pi-aa
                bb = aa/np.pi
                color = (bb,bb/10,bb)
#                if phi_fin>np.pi/2:
#                    color = (1,0,0)
#                else:
#                    color = 'k'
                ax.arrow(x - l/2*np.sin(phi_fin),y - l/2*np.cos(phi_fin),l*np.sin(phi_fin), l*np.cos(phi_fin),head_width=hw,head_length=hl,color=color,lw=0.5)
        ax.set_xlim(-abs(a1_m[0])/4*3,abs(a1_m[0])/4*3)
        ax.set_ylim(-abs(a1_m[0])/4*3,abs(a1_m[0])/4*3)
#        ax.set_ylim(-abs(a2_m[1])/4*3,abs(a2_m[1])/4*3)
#    if "title" in kwargs:
#        fig.suptitle(kwargs['title'],size=20)
    fig.tight_layout()
    if "save_figname" in kwargs and not kwargs['machine']=='loc':
        plt.savefig('results/figures/'+kwargs['save_figname']+'.png')
    else:
        plt.show()

def dist_xy(x,y):
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

def inside_UC(a,b,mi,qi,a1_m,a2_m,a12_m):
    x = np.array([a,b])
    if dist_xy(x,a2_m)<np.linalg.norm(x) and dist_xy(x,a2_m)<dist_xy(x,a12_m):
        return a-a2_m[0],b-a2_m[1]
    elif dist_xy(x,a1_m)<np.linalg.norm(x) and dist_xy(x,a1_m)<dist_xy(x,a12_m):
        return a-a1_m[0],b-a1_m[1]
    elif dist_xy(x,a12_m)<np.linalg.norm(x):
        return a-a1_m[0]-a2_m[0],b-a1_m[1]-a2_m[1]
    else:
        return a,b

def line(x,args,s=1):
    xi,yi,mx,my = args
    mx *= s
    my *= s
    if type(x) == np.float64:
        return np.array([x*xi+mx,x*yi+my])
    res = np.zeros((len(x),2))
    for i in range(len(x)):
        res[i] = np.array([x[i]*xi+mx,x[i]*yi+my])
    return res

def get_BZ_args(a1_m,a2_m,a12_m):
    if a1_m[1] == 0:
        x_1 = a2_m[1]/2+a2_m[0]/2*a12_m[0]/a12_m[1]
        args_1 = (0,1,a1_m[0]/2,a1_m[1]/2)
        x_2 = a2_m[1]*(a1_m[0]*a12_m[0]+a1_m[1]*a12_m[1])/2/(a12_m[0]*a2_m[1]-a12_m[1]*a2_m[0])
        args_2 = (1,-a2_m[0]/a2_m[1],a2_m[0]/2,a2_m[1]/2)
        x_12 = -a2_m[0]/2
        args_12 = (1,-a12_m[0]/a12_m[1],a12_m[0]/2,a12_m[1]/2)
    elif a2_m[1] == 0:
        x_1 = a1_m[1]*(a2_m[0]*a12_m[0]+a2_m[1]*a12_m[1])/2/(a12_m[0]*a1_m[1]-a12_m[1]*a1_m[0])
        args_1 = (1,-a1_m[0]/a1_m[1],a1_m[0]/2,a1_m[1]/2)
        x_2 = a1_m[1]/2+a1_m[0]/2*a12_m[0]/a12_m[1]
        args_2 = (0,1,a2_m[0]/2,a2_m[1]/2)
        x_12 = -a1_m[0]/2
        args_12 = (1,-a12_m[0]/a12_m[1],a12_m[0]/2,a12_m[1]/2)
    elif a12_m[1] == 0:
        x_1 = a1_m[1]*(a2_m[0]*a12_m[0]+a2_m[1]*a12_m[1])/2/(a12_m[0]*a1_m[1]-a12_m[1]*a1_m[0])
        args_1 = (1,-a1_m[0]/a1_m[1],a1_m[0]/2,a1_m[1]/2)
        x_2 = a2_m[1]*(a1_m[0]*a12_m[0]+a1_m[1]*a12_m[1])/2/(a12_m[0]*a2_m[1]-a12_m[1]*a2_m[0])
        args_2 = (1,-a2_m[0]/a2_m[1],a2_m[0]/2,a2_m[1]/2)
        x_12 = -a2_m[0]/2*a1_m[0]/a1_m[1]-a2_m[1]/2
        args_12 = (0,1,a12_m[0]/2,a12_m[1]/2)
    else:
        x_1 = a1_m[1]*(a2_m[0]*a12_m[0]+a2_m[1]*a12_m[1])/2/(a12_m[0]*a1_m[1]-a12_m[1]*a1_m[0])
        args_1 = (1,-a1_m[0]/a1_m[1],a1_m[0]/2,a1_m[1]/2)
        x_2 = a2_m[1]*(a1_m[0]*a12_m[0]+a1_m[1]*a12_m[1])/2/(a12_m[0]*a2_m[1]-a12_m[1]*a2_m[0])
        args_2 = (1,-a2_m[0]/a2_m[1],a2_m[0]/2,a2_m[1]/2)
        x_12 = x_1-a2_m[0]/2
        args_12 = (1,-a12_m[0]/a12_m[1],a12_m[0]/2,a12_m[1]/2)
    return [x_1,x_2,x_12],[args_1,args_2,args_12]

def get_l_args(args_i):
    mi = []
    qi = []
    for i in range(3):
        mi.append((args_i[i][1])/(args_i[i][0]) if not args_i[i][0]==0 else 0)
        qi.append(args_i[i][3] - mi[i]*args_i[i][2])
    return mi,qi

def plot_phis(phi,A_M,txt_title='mah'):
    """Plot the phases phi_1 and phi_2 in a 3D graph

    Parameters
    ----------
    phi: 2-tuple
        Symmetric and Anti-Symmetric phases.
    txt_title : string (optional)
        Title of the plot.

    """
    gx,gy = phi[0].shape
    X,Y = np.meshgrid(np.linspace(0,np.linalg.norm(A_M[0]),gx,endpoint=False),np.linspace(0,np.linalg.norm(A_M[1]),gy,endpoint=False))
    #
    fig = plt.figure(figsize=(20,10))
    plt.suptitle(txt_title)
    nn = len(phi)
    col = 3 if nn>=3 else nn
    for i in range(nn):
        ax = fig.add_subplot(nn//3+1,col,i+1,projection='3d')
        surf = ax.plot_surface(X, Y, phi[i].T, cmap=cm.coolwarm,
                   linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('x')
    plt.show()

def plot_Phi(Phi,a1_m,a2_m,title=''):
    """Plot interlayer.

    Parameters
    ----------
    Phi : ndarray
        Interlayer poptential.
    title : string, optional (default = '').
        Plot title.
    """
    gg = Phi.shape[0]
    X,Y = np.meshgrid(np.linspace(-1,2,3*gg),np.linspace(-1,2,3*gg))
    A1 = X*a1_m[0] + Y*a2_m[0]
    A2 = X*a1_m[1] + Y*a2_m[1]
    big_Phi = np.zeros((3*gg,3*gg))
    for i_ in range(3):
        for j_ in range(3):
            big_Phi[i_*gg:(i_+1)*gg,j_*gg:(j_+1)*gg] = Phi
    #
    fig, ax = plt.subplots(figsize=(13,8))
    ax.axis('off')
    ax.set_aspect(1.)
    con = ax.contour(A1,A2,big_Phi.T,
            levels=[0,],
            colors=('r',),
            linestyles=('-',),
            linewidths=(0.5,),
            ) 
    #Interlayer
    surf = ax.contourf(A1,A2,big_Phi.T,levels=100,cmap='RdBu')
    plt.colorbar(surf)
    ax.arrow(0,0,a1_m[0],a1_m[1])
    ax.arrow(0,0,a2_m[0],a2_m[1])
    ax.set_xlim(min([a1_m[0],a2_m[0],0])*2,max([a1_m[0],a2_m[0],0])*2)
    ax.set_ylim(min([a1_m[1],a2_m[1],0])*2,max([a1_m[1],a2_m[1],0])*2)
    ax.scatter(0,0)
    #Vectors
    fig.tight_layout()
    plt.show()
    exit()

def extend(phi,nn):
    """Extend the domain of phi from 0,A_M to -A_M,2*A_M by copying it periodically.

    Parameters
    ----------
    phi: np.ndarray
        Function on a grid to extend.
    nn: int
        Number of copies per direction.

    Returns
    -------
    np.ndarray
        Values of phi periodically repeated on larger domain.
    """
    gx,gy = phi.shape
    L = np.zeros((nn*gx,nn*gy),dtype=type(phi[0,0]))
    for i in range(nn):
        for j in range(nn):
            L[i*gx:(i+1)*gx,j*gy:(j+1)*gy] = phi
    return L

def reshape_Phi(phi,xp,yp):
    try:
        linx = np.linspace(0,1,phi.shape[0])
        liny = np.linspace(0,1,phi.shape[1])
        fun = RBS(linx,liny,phi)
        linx = np.linspace(0,1,xp)
        liny = np.linspace(0,1,yp)
        #X,Y = np.meshgrid(linx,liny)
        return fun(linx,liny)
    except: #Uniaxial ni=0 case
        linx = np.linspace(0,1,phi.shape[0])
        from scipy.interpolate import interp1d
        fun = interp1d(linx,phi)
        linx = np.linspace(0,1,xp)
        new_x = fun(linx)
        new_Phi = np.zeros((xp,yp))
        for i in range(yp):
            new_Phi[:,i] = new_x
        return new_Phi

def get_gridsize(max_grid,a1_m,a2_m):
    l_g = np.zeros(2,dtype=int)
    n_m = np.array([np.linalg.norm(a1_m),np.linalg.norm(a2_m)])
    i_m = np.argmax(n_m)
    l_g[i_m] = max_grid
    if abs(n_m[1-i_m]-1)<1e-5:
        l_g[1-i_m] = 3
    else:
        l_g[1-i_m] = int(max_grid/n_m[i_m]*n_m[1-i_m])
    if l_g[1-i_m] < 3:
        l_g[1-i_m] = 3
    return l_g

def get_fig_dn(machine):
    """Computes the directory name where to save the interlayer potential.

    Parameters
    ----------
    cluster: bool, optional
        Wether we are in the cluster or not (default is 'loc').

    Returns
    -------
    string
        The directory name.
    """
    return get_res_dn(machine)+'figures/'

def get_fig_pd_fn(moire_pars,grid_pts,gamma,machine):
    moire_dn = get_moire_dn(moire_pars,grid_pts,machine)[:-1]
    return get_fig_dn(machine) + 'PD_' + moire_dn[len(moire_dn)-moire_dn[::-1].index('/'):] +'_'+gamma+'.png'

def get_fig_mp_fn(moire_pars,grid_pts,rho,ani,machine):
    moire_dn = get_moire_dn(moire_pars,grid_pts,machine)[:-1]
    return get_fig_dn(machine) + 'MP_' + moire_dn[len(moire_dn)-moire_dn[::-1].index('/'):] +'_'+rho+'_'+ani+'.png'

def get_hdf5_dn(machine):
    """Computes the directory name where to save the interlayer potential.

    Parameters
    ----------
    cluster: bool, optional
        Wether we are in the cluster or not (default is 'loc').

    Returns
    -------
    string
        The directory name.
    """
    return get_res_dn(machine)+'hdf5/'

def get_hdf5_fn(moire_pars,grid_pts,machine):
    moire_dn = get_moire_dn(moire_pars,grid_pts,machine)[:-1]
    return get_hdf5_dn(machine) + moire_dn[len(moire_dn)-moire_dn[::-1].index('/'):] + '.hdf5'

def get_hdf5_par_fn(moire_pars,grid_pts,machine):
    moire_dn = get_moire_dn(moire_pars,grid_pts,machine)[:-1]
    return get_hdf5_dn(machine) + 'par_' + moire_dn[len(moire_dn)-moire_dn[::-1].index('/'):] + '.hdf5'

def get_pd_dn(machine):
    """Computes the directory name where to save the interlayer potential.

    Parameters
    ----------
    cluster: bool, optional
        Wether we are in the cluster or not (default is 'loc').

    Returns
    -------
    string
        The directory name.
    """
    return get_res_dn(machine) +'phase_diagrams_data/'

def get_moire_dn(moire_pars,grid_pts,machine):
    gx,gy = grid_pts
    return get_pd_dn(machine) + moire_pars_fn(moire_pars)+'_'+str(gx)+'x'+str(gy)+'/'

def get_gamma_dn(moire_pars,grid_pts,gamma,machine):
    return get_moire_dn(moire_pars,grid_pts,machine) + 'gamma_'+"{:.4f}".format(gamma)+'/'

def get_sol_fn(moire_pars,grid_pts,phys_args,machine):
    """Computes the filename of the interlayer coupling.

    Parameters
    ----------
    cluster: bool, optional
        Wether we are in the cluster or not (default is 'loc').

    Returns
    -------
    string
        The name of the .npy file containing the interlayer coupling.
    """
    gamma,rho,anisotropy = phys_args
    return get_gamma_dn(moire_pars,grid_pts,gamma,machine)+'sol_'+"{:.5f}".format(rho)+'_'+"{:.5f}".format(anisotropy)+'.npy'

def get_Phi_dn(machine):
    """Computes the directory name where to save the interlayer potential.

    Parameters
    ----------
    cluster: bool, optional
        Wether we are in the cluster or not (default is 'loc').

    Returns
    -------
    string
        The directory name.
    """
    return get_res_dn(machine) +'Phi_values/'

def get_Phi_fn(moire_pars,machine,rescaled):
    """Computes the filename of the interlayer coupling.

    Parameters
    ----------
    cluster: bool, optional
        Wether we are in the cluster or not (default is 'loc').

    Returns
    -------
    string
        The name of the .npy file containing the interlayer coupling.
    """
    txt_rs = 'rescaled' if rescaled else 'original'
    return get_Phi_dn(machine) + 'Phi_'+moire_pars_fn(moire_pars)+'_'+txt_rs+'.npy'

def get_AM_fn(moire_pars,machine):
    """Computes the filename of the interlayer coupling.

    Parameters
    ----------
    cluster: bool, optional
        Wether we are in the cluster or not (default is 'loc').

    Returns
    -------
    string
        The name of the .npy file containing the interlayer coupling.
    """
    return get_Phi_dn(machine) + 'AM_'+moire_pars_fn(moire_pars)+'.npy'

def moire_pars_fn(dic):
    """Generates a filename with the parameters formatted accordingly and a given extension.

    """
    fn = ''
    for k in dic.keys():
        fn += k+':'
        if type(dic[k])==type('string'):
            fn += dic[k]
        elif type(dic[k])==type(1) or type(dic[k])==np.int64:
            fn += str(dic[k])
        elif type(dic[k])==type(1.1) or type(dic[k])==np.float64:
            fn += "{:.4f}".format(dic[k])
        else:
            print("Parameter ",dic[k]," has unknown data type ",type(dic[k]))
            exit()
        fn += '_'
    return fn[:-1]

def get_res_dn(machine):
    """Computes the directory name where to save the interlayer potential.

    Parameters
    ----------
    cluster: bool, optional
        Wether we are in the cluster or not (default is 'loc').

    Returns
    -------
    string
        The directory name.
    """
    return get_home_dn(machine)+'results/'

def get_home_dn(machine):
    if machine == 'loc':
        return '/home/dario/Desktop/git/CrBr3/2_magnetization_plots/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/2_magnetization_plots/'
    elif machine == 'maf':
        pass

def get_machine(pwd):
    """Selects the machine the code is running on by looking at the working directory. Supports local, hpc (baobab or yggdrasil) and mafalda.

    Parameters
    ----------
    pwd : string
        Result of os.pwd(), the working directory.

    Returns
    -------
    string
        An acronim for the computing machine.
    """
    if pwd[6:11] == 'dario':
        return 'loc'
    elif pwd[:20] == '/home/users/r/rossid':
        return 'hpc'
    elif pwd[:13] == '/users/rossid':
        return 'maf'

def R_z(t):
    """Z-rotations of angle t.

    Parameters
    ----------
    t : float
        Rotation angle.

    Returns
    -------
    ndarray
        2x2 matrix.
    """
    R = np.zeros((2,2))
    R[0,0] = np.cos(t)
    R[0,1] = -np.sin(t)
    R[1,0] = np.sin(t)
    R[1,1] = np.cos(t)
    return R

def Moire(moire_pars,machine,rescaled):
    """The Moire script as a function
    """
    #
    disp = (machine=='loc')
    Phi_fn = get_Phi_fn(moire_pars,machine,rescaled)
    AM_fn = get_AM_fn(moire_pars,machine)
    #Impoert DFT data -> rescaled or not
    I = get_dft_data(machine,rescaled)
    #Interpolate interlayer DFT data
    pts = I.shape[0]
    big_I = extend(I,5)
    S_array = np.linspace(-2,3,5*pts,endpoint=False)
    fun_I = RBS(S_array,S_array,big_I)
    if moire_pars['type']=='const': #Sufficient to save just the values
        val = float(fun_I(0,0)) if moire_pars['place']=='AA' else float(fun_I(1/3,0))
        np.save(Phi_fn,np.ones(I.shape)*val)
        return 0
    #Lattice-1 and lattice-2
    l1_t,l2_t,a1_t,a2_t = compute_lattices(moire_pars)
    #Chose moire vectors 1 and 2 depending on x component
    if a1_t[0]>a2_t[0]:
        a1_m = a1_t
        a2_m = a2_t
        l1 = l1_t
        l2 = l2_t
    else:
        a1_m = a2_t
        a2_m = a1_t
        l1 = l2_t
        l2 = l1_t
    #Save moire vectors
    np.save(AM_fn,np.array([a1_m,a2_m]))
    if 0 and disp:   #Plot Moirè pattern
        fig,ax = plt.subplots(figsize=(20,20))
        ax.set_aspect('equal')
        #
        sss = max(np.linalg.norm(a1_m),np.linalg.norm(a2_m))
        for n in range(2):      #sublattice index
            for y in range(l1.shape[1]):
                ax.scatter(l1[:,y,n,0],l1[:,y,n,1],color='b',s=sss/50)
                ax.scatter(l2[:,y,n,0],l2[:,y,n,1],color='r',s=sss/50)

        ax.arrow(0,0,a1_m[0],a1_m[1],color='k',lw=2,head_width=0.5)
        ax.arrow(0,0,a2_m[0],a2_m[1],color='k',lw=2,head_width=0.5)
        ax.axis('off')
        plt.show()
    if moire_pars['type']=='biaxial':
        #Biaxial is actially the same as original interlayer coupling, with different moire vectors
        np.save(Phi_fn,I)
        return 0
    if moire_pars['type']=='uniaxial':
        if moire_pars['ni']==0 and 0:
            uni_I = I[:,int(moire_pars['tr']*I.shape[1])]
            np.save(Phi_fn,uni_I)
            return 0
        else:
            np.save(Phi_fn,I)
            return 0
    #Compute interlayer energy by evaluating the local stacking of the two layers
    xpts = ypts = 200 #if machine == 'loc' else 400
    J = np.zeros((xpts,ypts))
    X = np.linspace(0,1,xpts,endpoint=False)
    Y = np.linspace(0,1,ypts,endpoint=False)
    for i in range(xpts):
        for j in range(ypts):     #Cycle over all considered points in Moirè unit cell
            site = X[i]*a1_m + Y[j]*a2_m    #x and y components of consider point
            x1,y1,UC = find_closest(l1,site,'nan')
            x2,y2,UC = find_closest(l2,site,UC)
            if i==j and 0:   #plot two lattices, chosen site and coloured closest sites
                plt.figure(figsize=(10,10))
                plt.gca().set_aspect('equal')
                for n in range(2):  #lattices
                    for y in range(l1.shape[1]):
                        plt.scatter(l1[:,y,n,0],l1[:,y,n,1],color='b',s=3)
                        plt.scatter(l2[:,y,n,0],l2[:,y,n,1],color='r',s=3)
                plt.scatter(l1[x1,y1,UC,0],l1[x1,y1,UC,1],color='g',s=15)
                plt.scatter(l2[x2,y2,UC,0],l2[x2,y2,UC,1],color='m',s=15)
                plt.scatter(site[0],site[1],color='b',s=20)
                plt.show()
            #Find displacement
            displacement = l1[x1,y1,UC] - l2[x2,y2,UC]
            S1 = displacement[0]+displacement[1]/np.sqrt(3)
            S2 = 2*displacement[1]/np.sqrt(3)
            #Find value of I[d] and assign it to J[x]
            J[i,j] = fun_I(S1,S2)
    #
    if disp:
        title = ""
        plot_Phi(J,a1_m,a2_m,title)
    np.save(Phi_fn,J)

def get_dft_data(machine,rescaled):
    txt_rs = 'rescaled' if rescaled else 'original'
    data_fn = get_home_dn(machine)+"Data/CrBr3_interlayer_"+txt_rs+".npy"
    if Path(data_fn).is_file():
        return np.load(data_fn)
    if rescaled:
        print("Need first to compute the rescaled version")
        exit()
    #Compute original one
    data_fn = get_home_dn(machine)+"Data/CrBr3_interlayer_original.npy"
    data_marco_fn = get_home_dn(machine)+"Data/CrBr3_scan.txt"
    with open(data_marco_fn,'r') as f:
        lines = f.readlines()
    #remove one every 2 lines -> empty
    l1 = []
    for i,l in enumerate(lines):
        if i%2==0:
            l1.append(l[:-1])
    #separate numbers
    data = np.zeros((len(l1),4))
    for i,l in enumerate(l1):
        a = l.split(' ')
        for j in range(4):
            data[i,j] = float(a[j])
    #Extract CrBr3 interlayer data in matrix form
    pts = int(np.sqrt(data.shape[0]))
    S_list = list(np.linspace(0,1,pts,endpoint=False))
    S_txt = []
    for s in S_list:
        S_txt.append("{:.5f}".format(s))
    I = np.zeros((pts,pts))
    for i in range(pts**2):
        x = "{:.5f}".format(data[i,0])
        y = "{:.5f}".format(data[i,1])
        ind1 = S_txt.index(x)
        ind2 = S_txt.index(y)
        I[ind1,ind2] = -(data[i,2]-data[i,3])/2/Spin**2
    #
    np.save(data_fn,I)
    return I

def compute_lattices(moire_pars):
    """Compute the 2 honeycomb lattices with lattice lengths A_1/A_2 and a twist angle theta.

    Parameters
    ----------
    A_1 : float
        Lattice length of lattice 1.
    A_2 : float
        Lattice length of lattice 2.
    theta : float
        Relative rotation of the 2 axes.

    Returns
    -------
    ndarray, ndarray, int, int
        Lattices 1 and 2, together with x and y extesions.
        
    """
    #Strain tensor
    if moire_pars['type']=='general':
        e_xx = moire_pars['e_xx']
        e_yy = moire_pars['e_yy']
        e_xy = moire_pars['e_xy']
        strain_tensor = np.array([[e_xx,e_xy],[e_xy,e_yy]])
    elif moire_pars['type']=='uniaxial':
        eps = moire_pars['eps']
        ni = moire_pars['ni']
        phi = moire_pars['phi']
        strain_tensor = np.matmul(np.matmul(R_z(-phi).T,np.array([[eps,0],[0,-ni*eps]])),R_z(-phi))
    elif moire_pars['type']=='biaxial':
        eps = moire_pars['eps']
        strain_tensor = np.identity(2)*eps
    elif moire_pars['type']=='shear':
        e_xy = moire_pars['e_xy']
        phi = moire_pars['phi']
        strain_tensor = np.matmul(np.matmul(R_z(-phi).T,np.array([[0,e_xy],[e_xy,0]])),R_z(-phi))
    #
    theta = moire_pars['theta']
    #Moire lattice vectors
    T = np.matmul(np.identity(2)+strain_tensor/2,R_z(-theta/2)) - np.matmul(np.identity(2)-strain_tensor/2,R_z(theta/2))
    try:
        #inv_T = np.linalg.inv(T)
        a1_m = np.matmul(np.linalg.inv(T).T,a1)  #Moire real space lattice vector 1
        a2_m = np.matmul(np.linalg.inv(T).T,a2)
        translation_2 = 0
    except: #ni=0 -> strain on lattice direction 1
        st_vec = np.copy(a1+0.1*a2)
        norm = np.linalg.norm(st_vec)
        st_vec = st_vec/norm
        a1_m = st_vec/T[0,0]
        a2_m = np.matmul(np.array([[0,-1],[1,0]]),st_vec)
        translation_2 = moire_pars['tr']
    n1_m = np.linalg.norm(a1_m)
    n2_m = np.linalg.norm(a2_m)
    Np = np.linalg.norm(a1_m+a2_m)
    Nm = np.linalg.norm(a1_m-a2_m)
    nnm = min(Np,Nm)
    if 0:   #change a_1/a_2 to the sum if the norm is lower than original vector
        if nnm < max(n1_m,n2_m):
            new_a = a1_m+a2_m if Np<Nm else a1_m-a2_m
            a1_m = new_a if n1_m>n2_m else a1_m
            a2_m = new_a if n2_m>n1_m else a2_m
    n1_m = np.linalg.norm(a1_m)
    n2_m = np.linalg.norm(a2_m)
    #
    A_M = max(n1_m,n2_m)
    n_x = n_y = 5       #number of moirè lenths to include in l1,l2
    xxx = int(n_x*A_M)
    yyy = int(n_y*A_M)
    l1 = np.zeros((xxx,yyy,2,2))
    l2 = np.zeros((xxx,yyy,2,2))
    a1_1 = np.matmul(np.identity(2)+strain_tensor/2,np.matmul(R_z(theta/2),a1)) #vector 1 of lattice 1
    a2_1 = np.matmul(np.identity(2)+strain_tensor/2,np.matmul(R_z(theta/2),a2)) #vector 2 of lattice 1
    offset_sublattice_1 = np.matmul(np.identity(2)+strain_tensor/2,np.matmul(R_z(theta/2),d_vec))
    a1_2 = np.matmul(np.identity(2)-strain_tensor/2,np.matmul(R_z(-theta/2),a1)) #vector 1 of lattice 1
    a2_2 = np.matmul(np.identity(2)-strain_tensor/2,np.matmul(R_z(-theta/2),a2)) #vector 2 of lattice 1
    offset_sublattice_2 = np.matmul(np.identity(2)-strain_tensor/2,np.matmul(R_z(-theta/2),d_vec))
    for i in range(xxx):
        for j in range(yyy):
            l1[i,j,0] = (i-n_x//2*A_M)*a1_1+(j-n_y//2*A_M)*a2_1
            l1[i,j,1] = l1[i,j,0] + offset_sublattice_1
            l2[i,j,0] = (i-n_x//2*A_M)*a1_2+(j-n_y//2*A_M)*a2_2
            l2[i,j,1] = l2[i,j,0] + offset_sublattice_2
    l2[:,:,:,0] += a2_m[0]*translation_2
    l2[:,:,:,1] += a2_m[1]*translation_2
    print("Moire lengths: ",n1_m,' ',n2_m)
    print("Angle (deg): ",180/np.pi*np.arccos(np.dot(a1_m/n1_m,a2_m/n2_m)))
    return l1,l2,a1_m,a2_m

def find_closest(lattice,site,UC_):
    """Finds the closest lattice site to the coordinates "site". 
    The lattice is stored in "lattice" and the search can be constrained 
    to the unit cell "UC_", if given.


    Parameters
    ----------
    lattice : ndarray
        Lattice sites storage. Lattice has shape nx,ny,2->unit cell index,2->x and y.
    site : np.array
        x and y coordinates of space.
    UC_ : string
        Unit cell to constrain the search.

    Returns
    -------
    int, int, UC
        Indexes of clusest site, together with corresponding UC.
    """
    X,Y,W,Z = lattice.shape
    #
    dist_A = np.sqrt((lattice[:,:,0,0]-site[0])**2+(lattice[:,:,0,1]-site[1])**2)
    dist_B = np.sqrt((lattice[:,:,1,0]-site[0])**2+(lattice[:,:,1,1]-site[1])**2)
    min_A = np.min(np.ravel(dist_A))
    min_B = np.min(np.ravel(dist_B))
    if UC_=='nan':
        if min_A < min_B:
            UC = 0
            arg = np.argmin(np.reshape(dist_A,(X*Y)))
        else:
            UC = 1
            arg = np.argmin(np.reshape(dist_B,(X*Y)))
    elif UC_==0:
        arg = np.argmin(np.reshape(dist_A,(X*Y)))
        UC = UC_
    elif UC_==1:
        arg = np.argmin(np.reshape(dist_B,(X*Y)))
        UC = UC_
    #Smallest y-difference in A and B sublattice
    argx = arg//Y
    argy = arg%Y
    if argx in [0,X-1] or argy in [0,Y-1]:
        print("Reached end of lattice, probably not good")
        exit()
    return argx,argy,UC

def check_directory(moire_pars,grid_pts,gamma,machine):
    #Phase diagrams dir
    pd_dn = get_pd_dn(machine)
    if not Path(pd_dn).is_dir():
        os.system('mkdir '+pd_dn)
    #Moire dir
    moire_dn = get_moire_dn(moire_pars,grid_pts,machine)
    if not Path(moire_dn).is_dir():
        os.system('mkdir '+moire_dn)
    #gamma dir -> contains the actual .npy results
    gamma_dn = get_gamma_dn(moire_pars,grid_pts,gamma,machine)
    if not Path(gamma_dn).is_dir():
        os.system('mkdir '+gamma_dn)
    #hdf5 dir
    hdf5_dn = get_hdf5_dn(machine)
    if not Path(hdf5_dn).is_dir():
        os.system('mkdir '+hdf5_dn)
    #figures dir
    fig_dn = get_fig_dn(machine)
    if not Path(fig_dn).is_dir():
        os.system('mkdir '+fig_dn)

def compute_magnetization(phi):
    """Computes the total magnetization of the 2 layers (max is 2), given phi which contains symmetric and antisymmetric phases.


    Parameters
    ----------
    phi: 2-tuple
        Symmetric and Anti-Symmetric phases.

    Returns
    -------
    float
        Total magnetization along z of the spin configuration.
    """
    gx,gy = phi[0].shape
    if phi.shape[0]==1:
        return np.nan
    #Single layer phases
    phi_1,phi_2 = phi
    total_magnetization = np.sum(np.cos(phi_1))/gx/gy + np.sum(np.cos(phi_2))/gx/gy
    return abs(total_magnetization)

def compute_magnetization_x(phi):
    """Computes the total magnetization of the 2 layers (max is 2), given phi which contains symmetric and antisymmetric phases.


    Parameters
    ----------
    phi: 2-tuple
        Symmetric and Anti-Symmetric phases.

    Returns
    -------
    float
        Total magnetization along z of the spin configuration.
    """
    gx,gy = phi[0].shape
    if phi.shape[0]==1:
        return np.nan
    #Single layer phases
    phi_1,phi_2 = phi
    total_magnetization = np.sum(np.sin(phi_1))/gx/gy + np.sum(np.sin(phi_2))/gx/gy
    return abs(total_magnetization)

def compute_compare_MPs(list_pars,figname,machine,ind=0):
    """Compute the magnetization plots.

    """
    s_ = 20
    if ind == 10:
        ind = 1
        square = True
        fig = plt.figure(figsize=(7,20))
    else:
        square = False
        fig = plt.figure(figsize=(20,20))
    for iii in range(len(list_pars)):
        #plt.subplot(4,4,iii+1)
        rho_str,ani_str,grid_pts,moire_pars,txt_name = list_pars[iii]
        hdf5_par_fn = get_hdf5_par_fn(moire_pars,grid_pts,machine)
        #Open and read h5py File
        data = []
        n = 0
        with h5py.File(hdf5_par_fn,'r') as f:
            for k in f.keys():
                if not k[:5] == 'gamma':
                    continue
                gamma_ = k[6:]           #6 fixed by len(gamma_)
                for p in f[k].keys():
                    rho_ = p[:p.index('_')]
                    ani_ = p[p.index('_')+1:]
                    if rho_ == rho_str and ani_ == ani_str:
                        data.append([float(gamma_),f[k][p][ind]])
                        n += 1
        if n == 0:  #No data here for some reason
            print("No data in ",hdf5_par_fn," for pars ",rho_str," and ",ani_str)
            continue
        M = np.array(data)
        if square:
            n_n = 2
            plt.plot(M[:,0],((M[:,1]-np.min(M[:,1]))/(np.max(M[:,1])-np.min(M[:,1])))**n_n,'-',marker='*',label=txt_name)
            #plt.plot(((M[:,1]-np.min(M[:,1]))/(np.max(M[:,1])-np.min(M[:,1])))**n_n,M[:,0],'-',marker='*',label=txt_name)
        else:
            plt.plot(M[:,0],M[:,1],'-',marker='*',label=txt_name)
    if square:
        pass
    else:
        plt.xlabel(r'$h_\bot(T)$',size=s_)
        list_yax = [r'$E$',r'$M_z$',r'$M_x$',r'$M_z^{AA}$',r'$M_x^{AA}$',r'$M_z^{M}$',r'$M_x^{M}$']
        plt.ylabel(list_yax[ind],size=s_)
        plt.legend(fontsize=s_)
    if machine == 'loc':
        plt.show()
    else:
        plt.savefig(figname)

def load_Moire(Phi_fn,moire_pars,machine):
    AM_fn = get_AM_fn(moire_pars,machine)
    for i in range(200):
        try:
            Phi = np.load(Phi_fn)
            if not moire_pars['type'] == 'const':
                a1_m,a2_m = np.load(AM_fn)
            else:
                a1_m = a1
                a2_m = a2
            return Phi,a1_m,a2_m
        except:
            print(i," let's try again")
    exit()

offset_solution = -0.1

def ts1_12(Phi,gx,gy):
    phi_1 = (np.sign(Phi+offset_solution)-1)*np.pi/4
    phi_2 = -(np.sign(Phi+offset_solution)-1)*np.pi/4
    return np.array([phi_1,phi_2])

def ts2_12(Phi,gx,gy):
    phi_1 = (np.sign(Phi+offset_solution)+1)*np.pi/4
    phi_2 = -(np.sign(Phi+offset_solution)+1)*np.pi/4 + np.pi
    return np.array([phi_1,phi_2])

def ta_12(Phi,gx,gy):
    phi_1 = (np.sign(Phi+offset_solution)-1)*np.pi/2
    phi_1 = -(Phi-np.max(Phi))/np.min(Phi-np.max(Phi))*np.pi
    phi_2 = np.zeros((gx,gy))
    return np.array([phi_1,phi_2])

custom_in_pt = (ts1_12,ts2_12,ta_12)





