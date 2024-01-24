import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams.update({"text.usetex": True,})
s_ = 20
import random

#Physical parameters
rho_phys = {'DFT':1.4,'exp':1.7} #     (meV)       #CHECK
d_phys = {'DFT':0.18,'exp':0.09} #     (meV)       #CHECK

#Interlayer potential which depends on Moirè pattern
#moire_type = 'general'
moire_type = 'uniaxial'
#moire_type = 'biaxial'
#moire_type = 'shear'

moire_pars = {
        'general':{
            'e_xx':0.1,
            'e_yy':0.3,
            'e_xy':0.15,
            },
        'uniaxial':{
            'eps':0.2,
            'ni':0.5,
            'phi':1,
            },
        'biaxial':{
            'eps':0.1,
            },
        'shear':{
            'e_xy':0.1,
            'phi':0,
            },

        'theta':0.,
        }

#Triangular lattice
a1 = np.array([1,0])
a2 = np.array([-1/2,np.sqrt(3)/2])
d = np.array([0,1/np.sqrt(3)])  #vector connecting the two sublattices
b1 = np.array([1,1/np.sqrt(3)])*2*np.pi
b2 = np.array([0,2/np.sqrt(3)])*2*np.pi

def compute_solution(gamma,args_m):
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
    rho,anisotropy = args_m['args_phys']
    gridx,gridy = args_m['grid']
    rg = args_m['pts_per_fit']
    #Variables for storing best solution
    min_E = 1e10
    result = np.zeros((2,gridx,gridy))
    for ind_in_pt in range(args_m['n_initial_pts']):
        E = []  #list of energies for the while loop
        if args_m['disp']:
            print("Starting minimization step ",str(ind_in_pt))
        #Initial condition -> just constant
        fs = ((ind_in_pt-1)//8)/4 if ind_in_pt<64 else random.random()
        fa = ((ind_in_pt-1)%8)/4 if ind_in_pt<64 else random.random()
        #Compute first state and energy
        phi = initial_point(fs,fa,gridx,gridy)
        d_phi = (compute_derivatives(phi[0],1,A_M,rg),compute_derivatives(phi[1],1,A_M,rg))
        E.append(compute_energy(phi,Phi,gamma,rho,anisotropy,d_phi))
        #Initiate learning rate and minimization loop
        step = 1        #initial step
        lr = args_m['learn_rate']
        while True:
            learn_rate = lr*random.random()
            #Energy gradients
            dHs = grad_H(phi,'s',Phi,gamma,rho,anisotropy,compute_derivatives(phi[0],2,A_M,rg))
            dHa = grad_H(phi,'a',Phi,gamma,rho,anisotropy,compute_derivatives(phi[1],2,A_M,rg))
            #Update phi
            phi[0] += learn_rate*dHs
            phi[1] += learn_rate*dHa
            #New energy
            d_phi = (compute_derivatives(phi[0],1,A_M,rg),compute_derivatives(phi[1],1,A_M,rg))
            E.insert(0,compute_energy(phi,Phi,gamma,rho,anisotropy,d_phi))
            #Check if dHs and dHa are very small
            if args_m['disp']:
                print("energy step ",step," is ",E[0])
            #Exit checks
            if check_energies(E):   #stable energy
                if E[0]<min_E:
                    min_E = E[0]
                    result = np.copy(phi)
                break
            if E[0]>E[1]:    #worse solution
                phi[0] -= learn_rate*dHs
                phi[1] -= learn_rate*dHa
                del E[0]
                lr /= 2
                if abs(lr) < 1e-7:
                    break
            #Max number of steps scenario
            if step > args_m['maxiter']:
                if ind_in_pt == 0:    #If this happens for the first minimization step, save a clearly fake one for later comparison
                    min_E = 1e8
                    result[0] = np.ones((gridx,gridy))*20
                    result[1] = np.ones((gridx,gridy))*20
                break
            step += 1
            #
        if args_m['disp']:
            print("Minimum energy at ",E[0])
            #plot_phis(phi,'phi_s and phi_a')
            plot_magnetization(phi,Phi,gamma,A_M)
    return result

def compute_energy(phi,Phi,gamma,rho,anisotropy,d_phi):
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
    #Old derivative squared
    gx,gy = phi[0].shape
    grad_2s = np.absolute(d_phi[0][0])**2 + np.absolute(d_phi[0][1])**2
    grad_2a = np.absolute(d_phi[1][0])**2 + np.absolute(d_phi[1][1])**2
    energy = rho/4*(grad_2s+grad_2a) - anisotropy*np.cos(phi[1])*np.cos(phi[0]) - Phi*np.cos(phi[1]) - 2*gamma*np.cos(phi[0]/2)*np.cos(phi[1]/2)
    H = energy.sum()/gx/gy
    return H

def grad_H(phi,tt,Phi,gamma,rho,anisotropy,d2_phi):
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
    res = -rho/2*(d2_phi[0]+d2_phi[1])
    if tt=='s':
        return res + anisotropy*np.sin(phi[0])*np.cos(phi[1]) + gamma*np.cos(phi[1]/2)*np.sin(phi[0]/2)
    elif tt=='a':
        return res + anisotropy*np.cos(phi[0])*np.sin(phi[1]) + Phi*np.sin(phi[1]) + gamma*np.cos(phi[0]/2)*np.sin(phi[1]/2)

def initial_point(fs,fa,gx,gy):
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
    phi_s = np.ones((gx,gy))*2*np.pi*fs
    phi_a = np.ones((gx,gy))*2*np.pi*fa
    return [phi_s, phi_a]

def compute_derivatives(phi,n,A_M,rg):
    """Compute the 'n' derivative of phi.

    Parameters
    ----------
    phi: np.ndarray
        Function on a grid.
    n: int
        Order of derivative to compute.

    Returns
    -------
    tuple
        2-tuple containing the x and y derivatives of order 'n' of phi. Only the second derivatives are smoothen out.
    """
    gridx,gridy = phi.shape
    xx = np.linspace(0,np.linalg.norm(A_M[0]),gridx,endpoint=False)
    yy = np.linspace(0,np.linalg.norm(A_M[1]),gridy,endpoint=False)
    #Interpolate phase
    fun = smooth(phi,rg,A_M)[1]
    #derivatives
    dn_phi_x = smooth(fun.partial_derivative(n,0)(xx,yy),rg,A_M)[0]
    dn_phi_y = smooth(fun.partial_derivative(0,n)(xx,yy),rg,A_M)[0]
    dn_phi = (dn_phi_x,dn_phi_y)
    return dn_phi

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
    nn = 5
    if len(list_E) <= nn:
        return False
    lim = 1e-8
    for i in range(nn):
        if abs(list_E[i]-list_E[i+1]) > lim:
            return False
    return True
#
def compute_lattices():
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
    if moire_type=='general':
        e_xx = moire_pars['general']['e_xx']
        e_yy = moire_pars['general']['e_yy']
        e_xy = moire_pars['general']['e_xy']
        strain_tensor = np.array([[e_xx,e_xy],[e_xy,e_yy]])
    elif moire_type=='uniaxial':
        eps = moire_pars['uniaxial']['eps']
        ni = moire_pars['uniaxial']['ni']
        phi = moire_pars['uniaxial']['phi']
        strain_tensor = np.matmul(np.matmul(R_z(-phi).T,np.array([[eps,0],[0,-ni*eps]])),R_z(-phi))
    elif moire_type=='biaxial':
        eps = moire_pars['biaxial']['eps']
        strain_tensor = np.identity(2)*eps
    elif moire_type=='shear':
        e_xy = moire_pars['shear']['e_xy']
        phi = moire_pars['shear']['phi']
        strain_tensor = np.matmul(np.matmul(R_z(-phi).T,np.array([[0,e_xy],[e_xy,0]])),R_z(-phi))
    #
    theta = moire_pars['theta']
    #Moire lattice vectors
    T = np.matmul(1+strain_tensor/2,R_z(-theta/2)) - np.matmul(1-strain_tensor/2,R_z(theta/2))
    b1_m = np.matmul(T,b1)  #Moire reciprocal lattice vector 1
    b2_m = np.matmul(T,b2)
    #There must be a better way...
    a1_m = np.array([0,0],dtype=float)
    if b2_m[1] == 0:
        a1_m[1] = 2*np.pi/b1_m[1]
    else:
        a1_m[0] = 2*np.pi/(b1_m[0]-b1_m[1]*b2_m[0]/b2_m[1])
        a1_m[1] = -a1_m[0]*b2_m[0]/b2_m[1]
    a2_m = np.array([0,0],dtype=float)
    if b1_m[1] == 0:
        a2_m[1] = 2*np.pi/b2_m[1]
    else:
        a2_m[0] = 2*np.pi/(b2_m[0]-b2_m[1]*b1_m[0]/b1_m[1])
        a2_m[1] = -a2_m[0]*b1_m[0]/b1_m[1]
    #
    A_M = np.linalg.norm(a2_m)
    n_x = n_y = 5       #number of moirè lenths to include in l1,l2
    xxx = int(n_x*A_M)
    yyy = int(n_y*A_M)
    l1 = np.zeros((xxx,yyy,2,2))
    l2 = np.zeros((xxx,yyy,2,2))
    a1_1 = np.matmul(np.identity(2)+strain_tensor/2,np.matmul(R_z(theta/2),a1)) #vector 1 of lattice 1
    a2_1 = np.matmul(np.identity(2)+strain_tensor/2,np.matmul(R_z(theta/2),a2)) #vector 2 of lattice 1
    offset_sublattice_1 = np.matmul(np.identity(2)+strain_tensor/2,np.matmul(R_z(theta/2),d))
    a1_2 = np.matmul(np.identity(2)-strain_tensor/2,np.matmul(R_z(-theta/2),a1)) #vector 1 of lattice 1
    a2_2 = np.matmul(np.identity(2)-strain_tensor/2,np.matmul(R_z(-theta/2),a2)) #vector 2 of lattice 1
    offset_sublattice_2 = np.matmul(np.identity(2)-strain_tensor/2,np.matmul(R_z(-theta/2),d))
    for i in range(xxx):
        for j in range(yyy):
            l1[i,j,0] = (i-n_x//2*A_M)*a1_1+(j-n_y//2*A_M)*a2_1
            l1[i,j,1] = l1[i,j,0] + offset_sublattice_1
            l2[i,j,0] = (i-n_x//2*A_M)*a1_2+(j-n_y//2*A_M)*a2_2
            l2[i,j,1] = l2[i,j,0] + offset_sublattice_2

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

def smooth(phi,rg,A_M):
    """Smooth out the periodic function phi.

    Parameters
    ----------
    phi: np.ndarray
        Function on a grid.

    Returns
    -------
    np.ndarray, function
        Values of phi after being smoothen by first extending it on a larger domain and then interpolate it on
        fewer points (second returned argument) and finally computed on the original grid.
    """
    gridx,gridy = phi.shape
    smooth_phi = np.zeros((gridx,gridy))
    for i in range(-rg,rg+1):
        for j in range(-rg,rg+1):
            smooth_phi += np.roll(np.roll(phi,i,axis=0),j,axis=1)
    smooth_phi /= (1+2*rg)**2
    xx = np.linspace(0,np.linalg.norm(A_M[0]),gridx,endpoint=False)
    yy = np.linspace(0,np.linalg.norm(A_M[1]),gridy,endpoint=False)
    fun = RBS(xx,yy,smooth_phi)
    return smooth_phi,fun

def get_dft_data(machine):
    data_fn = get_home_dn(machine)+"Data/CrBr3_interlayer.npy"
    if Path(data_fn).is_file():
        return np.load(data_fn)
    #Compute it
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
        I[ind1,ind2] = -(data[i,2]-data[i,3])/2
    #
    np.save(data_fn,I)
    return I

def plot_magnetization(phi,Phi,gamma,A_M,save=False,tt=''):
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
    #Interpolate Phi
    XX = np.linspace(-1,2,3*gx,endpoint=False)
    YY = np.linspace(-1,2,3*gy,endpoint=False)
    big_Phi = extend(Phi,3)
    fun_Phi = RBS(XX,YY,big_Phi)
    #Single layer phases
    phi_1 = (phi[0]+phi[1])/2
    phi_2 = (phi[0]-phi[1])/2
    #Background -> interlayer coupling
    long_X = np.linspace(-1,1,2*gx,endpoint=False)
    long_Y = np.linspace(-1,1,2*gy,endpoint=False)
    X,Y = np.meshgrid(long_X,long_Y)
    A1 = X*A_M[0][0] + Y*A_M[1][0]
    A2 = X*A_M[0][1] + Y*A_M[1][1]
    #Box the Moirè unit cell
#    s3 = np.sqrt(3)
#    def line(x,y0,q):
#        return y0+q*x
#    xx14 = np.linspace(-1/2,0,100)
#    xx23 = np.linspace(0,1/2,100)
#    pars = [[(1/s3,1/s3),(-1/s3,-1/s3)],
#            [(1/s3,-1/s3),(-1/s3,1/s3)]]
    #Plot the arrows
    l = 0.02       #length of arrow
    hw = 0.01       #arrow head width
    hl = 0.01       #arrow head length
    facx = gx//20     #plot 1 spin every "fac" of grid
    facy = gy//20     #plot 1 spin every "fac" of grid
    def inside_UC(a,b):

        return a,b
    phi_ = [phi_1,phi_2]
    #Plot magnetization patterns -> two different plots
    fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(20,10))
    for ind,ax in enumerate([ax1,ax2]):
        ax.axis('off')
        ax.set_aspect(1.)
        ax.contour(A1,A2,fun_Phi(long_X,long_Y).T,levels=[0,],colors=('r',),linestyles=('-',),linewidths=(1,))
        surf = ax.contourf(A1,A2,fun_Phi(long_X,long_Y).T,levels=20)
#        ax.vlines(1/2,-1/2/s3,1/2/s3,linestyles='dashed',color='r')
#        ax.vlines(-1/2,-1/2/s3,1/2/s3,linestyles='dashed',color='r')
#        for i,x in enumerate([xx14,xx23]):
#            for j in range(2):
#                ax.plot(x,line(x,*pars[i][j]),linestyle='dashed',color='r')
        for i in range(gx//facx):
            for j in range(gy//facy):
                x_c = (i*facx)/gx
                y_c = (j*facy)/gy
                x = x_c*A_M[0][0] + y_c*A_M[1][0]
                y = x_c*A_M[0][1] + y_c*A_M[1][1]
                x,y = inside_UC(x,y)
                phi_fin = phi_[ind][i*facx,j*facy]
                ax.arrow(x - l/2*np.sin(phi_fin),y - l/2*np.cos(phi_fin),l*np.sin(phi_fin), l*np.cos(phi_fin),head_width=hw,head_length=hl,color='k')
#        ax.set_xlim(-0.6,0.6)
#        ax.set_ylim(-0.65,0.65)
    plt.suptitle("gamma = "+"{:.4f}".format(gamma),size=20)
    plt.show()

def plot_phis(phi,txt_title='mah'):
    """Plot the phases phi_1 and phi_2 in a 3D graph

    Parameters
    ----------
    phi: 2-tuple
        Symmetric and Anti-Symmetric phases.
    txt_title : string (optional)
        Title of the plot.

    """
    gx,gy = phi[0].shape
    X,Y = np.meshgrid(np.linspace(0,1,gx,endpoint=False),np.linspace(0,1,gy,endpoint=False))
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
    plt.show()
    exit()

def plot_Phi(Phi,a1_m,a2_m,title=''):
    """Plot interlayer.

    Parameters
    ----------
    Phi : ndarray
        Interlayer poptential.
    title : string, optional (default = '').
        Plot title.
    """
    gridx,gridy = Phi.shape
    lx = np.linspace(0,1,gridx)
    ly = np.linspace(0,1,gridy)
    X,Y = np.meshgrid(lx,ly)
    A1 = X*a1_m[0] + Y*a2_m[0]
    A2 = X*a1_m[1] + Y*a2_m[1]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.axis('off')
    ax.set_aspect(1.)
    ax.contour(A1,A2,Phi.T,levels=[0,],colors=('r',),linestyles=('-',),linewidths=(0.5))
    surf = ax.contourf(A1,A2,Phi.T,levels=20)
    plt.show()

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
    L = np.zeros((nn*gx,nn*gy))
    for i in range(nn):
        for j in range(nn):
            L[i*gx:(i+1)*gx,j*gy:(j+1)*gy] = phi
    return L

def reshape_Phi(phi,xp,yp):
    linx = np.linspace(0,1,phi.shape[0])
    liny = np.linspace(0,1,phi.shape[1])
    fun = RBS(linx,liny,phi)
    linx = np.linspace(0,1,xp)
    liny = np.linspace(0,1,yp)
    #X,Y = np.meshgrid(linx,liny)
    return fun(linx,liny)

def get_sol_fn(gamma,machine):
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
    return get_Phi_dn(machine) + 'sol_'+"{:.4f}".format(gamma)+'_'+moire_type+'_'+general_fn([*moire_pars[moire_type].values()],'.hdf5')

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
    return get_home_dn(machine)+'Phi_values/'

def get_Phi_fn(machine):
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
    return get_Phi_dn(machine) + 'Phi_'+moire_type+'_'+general_fn([*moire_pars[moire_type].values()],'.hdf5')

def general_fn(pars,extension):
    """Generates a filename with the parameters formatted accordingly and a given extension.

    """
    fn = ''
    for i,p in enumerate(pars):
        if type(p)==type('string'):
            fn += p
        elif type(p)==type(1):
            fn += str(p)
        elif type(p)==type(1.1):
            fn += "{:.4f}".format(p)
        else:
            print("Parameter ",p," has unknown data type ",type(p))
            exit()
        if not i==len(pars)-1:
            fn += '_'
    fn += extension
    return fn

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
