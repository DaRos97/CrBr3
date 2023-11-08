import numpy as np
import random

#Triangular lattice
a1 = np.array([1,0])
a2 = np.array([-1/2,np.sqrt(3)/2])
b1 = np.array([1,1/np.sqrt(3)])*2*np.pi
b2 = np.array([0,2/np.sqrt(3)])*2*np.pi
#Square lattice
#a1 = np.array([1,0]) 
#a2 = np.array([0,1]) 
#b1 = np.array([1,0])*2*np.pi
#b2 = np.array([0,1])*2*np.pi

def find_closest(lattice,site,UC_):
    #Finds the closest lattice site to the coordinates "site". The lattice is stored in "lattice" 
    #and the search can be constrained to the unit cell "UC_", if given.

    #Lattice has shape nx,ny,2->unit cell index,2->x and y.
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
    if argx == X-1 or argy == Y-1 or argx == 0 or argy == 0:
        print("Reached end of lattice, probably not good")
        exit()
    return argx,argy,UC

def compute_lattices(A_1,A_2,theta):
    A_M = moire_length(A_1,A_2,theta)
    n_x = 3
    n_y = 3
    xxx = int(2*n_x*A_M)
    yyy = int(2*n_y*A_M)
    l1 = np.zeros((xxx,yyy,2,2))
    l2 = np.zeros((xxx,yyy,2,2))
    a1_1 = np.matmul(R_z(theta/2),a1)*A_1
    a2_1 = np.matmul(R_z(theta/2),a2)*A_1
    offset_sublattice_1 = np.matmul(R_z(theta/2),np.array([0,A_1/np.sqrt(3)]))
    a1_2 = np.matmul(R_z(-theta/2),a1)*A_2
    a2_2 = np.matmul(R_z(-theta/2),a2)*A_2
    offset_sublattice_2 = np.matmul(R_z(-theta/2),np.array([0,A_2/np.sqrt(3)]))
    for i in range(xxx):
        for j in range(yyy):
            l1[i,j,0] = (i-n_x*A_M)*a1_1+(j-n_y*A_M)*a2_1
            l1[i,j,1] = l1[i,j,0] + offset_sublattice_1
            l2[i,j,0] = (i-n_x*A_M)*a1_2+(j-n_y*A_M)*a2_2
            l2[i,j,1] = l2[i,j,0] + offset_sublattice_2
    return l1,l2,xxx,yyy

def moire_length(A_1,A_2,theta):
    if A_1 == 1 and A_2 == 1 and theta == 0:
        return 1
    return 1/np.sqrt(1/A_1**2+1/A_2**2-2*np.cos(theta)/(A_1*A_2))

def R_z(t):
    R = np.zeros((2,2))
    R[0,0] = np.cos(t)
    R[0,1] = -np.sin(t)
    R[1,0] = np.sin(t)
    R[1,1] = np.cos(t)
    return R

###########################################################################################
###########################################################################################
###########################################################################################

def compute_grid_pd(pts_array):
    """Computes the grid of alpha/beta points to consider in order to build up the phase diagram.
    The phase diagram is computed in scale of a/(1+a), so we consider pts_array values 'g' between 
    0 and 1 and compute alpha/beta as alpha(beta)=1/(1-g).

    Parameters
    ----------
    pts_array : int
        The number of elements to consider in each direction, alpha and beta.

    Returns
    -------
    np.ndarray
        a matrix of values of size (pts_array,pts_array,2).
    """
    values = np.zeros((pts_array,pts_array,2))
    g_array = np.linspace(0,1,20,endpoint=False)
    for i in range(pts_array):
        for j in range(pts_array):
            values[i,j,0] = g_array[i]/(1-g_array[i])
            values[i,j,1] = g_array[j]/(1-g_array[j])
    return values


def initial_point(Phi,alpha,beta,grid,fs,fa,ans):
    """Computes the initial point for the minimization. The possibilities are for now
    either twisted-s -> ans=0, or constant -> ans=1

    Parameters
    ----------
    Phi : np.ndarray
        Interlayer coupling of size (grid,grid)
    alpha: float
        Parameter alpha.
    beta: float
        Parameter beta.
    grid : int
        The number of points in each direction.
    fs : float
        Random number between 0 and 1 to set inisial condition for symmetric phase.
    fa : float
        Random number between 0 and 1 to set inisial condition for a-symmetric phase.
    ans : int
        Index to chose the ansatz of the initial point.

    Returns
    -------
    tuple
        Symmetric and antisymmetric phases at each position (grid,grid) of the Moirè unit cell.
    """
    initial_ansatz = ['twisted-s','constant']
    sol = initial_ansatz[ans]
    if sol=='twisted-s':      #ansatz for alpha,beta<<1
        delta = beta/alpha**2
        if abs(delta)>3/2:
            print("delta ",str(delta)," too large for twisted-s, switching to constant initial condition")
            return initial_point(Phi,alpha,beta,grid,fs,fa,1)
        phi0 = np.arccos(2/3*delta)
        const = 1/2-np.tan(phi0)**(-2)
        phi_s = np.ones((grid,grid))*np.pi
        phi_a = phi0 - alpha*np.sin(phi0)*(Phi-const)
    elif sol=='constant':
        phi_s = np.ones((grid,grid))*2*np.pi*fs
        phi_a = np.ones((grid,grid))*2*np.pi*fa
    return phi_s, phi_a

def compute_energy(phi_s,phi_a,Phi,alpha,beta,grid,A_M):
    """Computes the energy of the system.

    Parameters
    ----------
    phi_s: np.ndarray
        Symmetric phases.
    phi_a: np.ndarray
        A-symmetric phases.
    Phi : np.ndarray
        Interlayer coupling of size (grid,grid)
    alpha: float
        Parameter alpha.
    beta: float
        Parameter beta.
    grid : int
        The number of points in each direction.
    A_M : float
        The Moire lattice length.

    Returns
    -------
    float
        Energy density summed over all sites.
    """
    dx = dy = A_M/grid
    grad_2s = np.absolute((np.roll(phi_s,-1,axis=0)-phi_s)/dx)**2 + np.absolute((np.roll(phi_s,-1,axis=1)-phi_s)/dy)**2
    grad_2a = np.absolute((np.roll(phi_a,-1,axis=0)-phi_a)/dx)**2 + np.absolute((np.roll(phi_a,-1,axis=1)-phi_a)/dy)**2
    kin_part = 1/2*(grad_2s+grad_2a)
    energy = kin_part - np.cos(phi_a)*(alpha*Phi+beta*np.cos(phi_s))
    H = energy.sum()/grid**2
    return H

def laplacian(phi,grid,A_M):
    """Computes the discrete Laplacian of 'phi'.

    Parameters
    ----------
    phi: np.ndarray
        Field to derivate
    grid : int
        The number of points in each direction.
    A_M : float
        The Moire lattice length.

    Returns
    -------
    np.ndarray
        Laplacian of 'phi' on the (grid,grid) space.
    """
    dx = dy = A_M/grid
    Dx = np.roll(phi,2,axis=0)-2*np.roll(phi,1,axis=0)+np.roll(phi,0,axis=0)
    Dy = np.roll(phi,-2,axis=1)-2*np.roll(phi,-1,axis=1)+np.roll(phi,0,axis=1)
    res = (Dx/dx**2 + Dy/dy**2)
    return res 
    
def grad_H(phi_s,phi_a,tt,Phi,alpha,beta,grid,A_M):
    """Computes evolution step dH/d phi.

    Parameters
    ----------
    phi_s: np.ndarray
        Symmetric phases.
    phi_a: np.ndarray
        A-symmetric phases.
    tt : char
        Determines which derivative to compute.
    Phi : np.ndarray
        Interlayer coupling of size (grid,grid)
    alpha: float
        Parameter alpha.
    beta: float
        Parameter beta.
    grid : int
        The number of points in each direction.
    A_M : float
        The Moire lattice length.

    Returns
    -------
    np.ndarray
        Gradient of Hamiltonian on the (grid,grid) space.
    """
    if tt=='s':
        return beta*np.sin(phi_s)*np.cos(phi_a) - laplacian(phi_s,grid,A_M)
    elif tt=='a':
        return (beta*np.cos(phi_s)+alpha*Phi)*np.sin(phi_a) - laplacian(phi_a,grid,A_M)

def compute_magnetization(Phi,alpha,beta,grid,A_M,args_minimization):
    """Computes the magnetization pattern by performing a gradient descent from random 
    initial points.

    Parameters
    ----------
    Phi : np.ndarray
        Interlayer coupling of size (grid,grid)
    alpha: float
        Parameter alpha.
    beta: float
        Parameter beta.
    grid : int
        The number of points in each direction.
    A_M : float
        The Moire lattice length.
    args_minimization : dic
        'rand_m' -> int, number of random initial seeds,
        'maxiter' -> int, max number of update evaluations.
        'disp' -> bool, diplay messages.

    Returns
    -------
    tuple
        Symmetric and antisymmetric phases at each position (grid,grid) of the Moirè unit cell.
    """
    if args_minimization['disp']:
        print("Parameters: alpha="+"{:.4f}".format(alpha)+", beta="+"{:.4f}".format(beta),'\n')
    #Initial condition parameterss
    ans = 1         #0->twisted-s, 1->const
    min_E = 0
    min_phi_s = np.zeros((grid,grid))
    min_phi_a = np.zeros((grid,grid))
    for sss in range(args_minimization['rand_m']):
        fs = random.random()
        fa = random.random()
        if sss==0:      #take pi/2,pi/2 as first guess
            fs = 0.5
            fa = 0.5
        #Define initial value of phi_s and phi_a
        phi_s,phi_a = initial_point(Phi,alpha,beta,grid,fs,fa,ans)
        E_0 = compute_energy(phi_s,phi_a,Phi,alpha,beta,grid,A_M)
        if args_minimization['disp']:
            print("Starting minimization step ",str(sss))
            print("starting energy: ",E_0)
        step = 1        #initial step
        lr_0 = -1       #standard learn rate
        learn_rate = lr_0
        while True:
            dHs = grad_H(phi_s,phi_a,'s',Phi,alpha,beta,grid,A_M)
            dHa = grad_H(phi_s,phi_a,'a',Phi,alpha,beta,grid,A_M)
            #
            phi_new_s = phi_s + learn_rate*dHs
            phi_new_a = phi_a + learn_rate*dHa
            #
            E_1 = compute_energy(phi_new_s,phi_new_a,Phi,alpha,beta,grid,A_M)
            phi_s = phi_new_s
            phi_a = phi_new_a
            conv_s = True if np.max(np.absolute(dHs)) < 1e-3 else False
            conv_a = True if np.max(np.absolute(dHa)) < 1e-3 else False
            if abs(E_0-E_1)<1e-10:
                final_E = E_1
                if args_minimization['disp']:
                    print("Initial guess: ",fs*np.pi,fa*np.pi," with final energy ",final_E,'\n')
                if final_E<min_E and conv_s and conv_a:
                    min_E = E_1
                    min_phi_s = phi_s
                    min_phi_a = phi_a
                break
            if E_1>E_0:     #going higher in energy -> go back and increase less the solution
                phi_s -= learn_rate*dHs
                phi_a -= learn_rate*dHa
                learn_rate *= 0.5
            else:
                E_0 = E_1
                learn_rate = lr_0
            #
            if step > args_minimization['maxiter']:
                if sss == 0:    #If this happens for the first minimization step, save a clearly fake one for later comparison
                    min_E = 1e8
                    min_phi_s = np.ones((grid,grid))*20
                    min_phi_a = np.ones((grid,grid))*20
                break
            step += 1
            #
    return min_phi_s, min_phi_a

def test_minimum(phi_s,phi_a,Phi,alpha,beta,grid,A_M):
    s = grad_H(phi_s,phi_a,'s',Phi,alpha,beta,grid,A_M)
    a = grad_H(phi_s,phi_a,'s',Phi,alpha,beta,grid,A_M)
    plot_phis(s,a,grid)

def plot_phis(phi_1,phi_2,grid):
    """Plot the phases phi_1 and phi_2 in a 3D graph

    Parameters
    ----------
    phi_1 : np.ndarray
        Phases defined on (grid,grid) space.
    phi_2 : np.ndarray
        Phases defined on (grid,grid) space.
    grid : int
        The number of points in each direction

    """
    import matplotlib.pyplot as plt 
    from matplotlib import cm
    #
    fig, (ax1,ax2) = plt.subplots(1,2,subplot_kw={"projection": "3d"},figsize=(20,10))
    X,Y = np.meshgrid(np.linspace(0,1,grid,endpoint=False),np.linspace(0,1,grid,endpoint=False))
    surf = ax1.plot_surface(X, Y, phi_1, cmap=cm.coolwarm,
               linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    surf = ax2.plot_surface(X, Y, phi_2, cmap=cm.coolwarm,
               linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_magnetization(phi_s,phi_a,Phi,grid):
    """Plots the magnetization values in the Moirè unit cell, with a background given by the
    interlayer potential. The two images correspond to the 2 layers. Magnetization is in x-z
    plane while the layers are in x-y plane.

    Parameters
    ----------
    phi_s : np.ndarray
        Symmetric phases defined on (grid,grid) space.
    phi_a : np.ndarray
        A-symmetric phases defined on (grid,grid) space.
    Phi : np.ndarray
        Interlayer coupling of size (grid,grid).
    grid : int
        The number of points in each direction

    """
    import matplotlib.pyplot as plt 
    from scipy.interpolate import RectBivariateSpline as RBS
    #Interpolate Phi
    XX = np.linspace(-2,2,4*grid,endpoint=False)
    big_J = np.zeros((4*grid,4*grid))
    big_J[:grid,:grid] = Phi; big_J[:grid,grid:2*grid] = Phi; big_J[:grid,2*grid:3*grid] = Phi; big_J[:grid,3*grid:] = Phi;
    big_J[grid:2*grid,:grid] = Phi; big_J[grid:2*grid,grid:2*grid] = Phi; big_J[grid:2*grid,2*grid:3*grid] = Phi; big_J[grid:2*grid,3*grid:] = Phi;
    big_J[2*grid:3*grid,:grid] = Phi; big_J[2*grid:3*grid,grid:2*grid] = Phi; big_J[2*grid:3*grid,2*grid:3*grid] = Phi; big_J[2*grid:3*grid,3*grid:] = Phi;
    big_J[3*grid:,:grid] = Phi; big_J[3*grid:,grid:2*grid] = Phi; big_J[3*grid:,2*grid:3*grid] = Phi; big_J[3*grid:,3*grid:] = Phi;
    fun_J = RBS(XX,XX,big_J)
    phi_1 = (phi_s+phi_a)/2
    phi_2 = (phi_s-phi_a)/2
    #Plot magnetization patterns -> two different plots
    fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(20,20))
    #Box the Moirè unit cell
    ax1.hlines(0,0,1,linestyles='dashed',color='r')
    ax1.hlines(1,0,1,linestyles='dashed',color='r')
    ax1.vlines(0,0,1,linestyles='dashed',color='r')
    ax1.vlines(1,0,1,linestyles='dashed',color='r')
    ax2.hlines(0,0,1,linestyles='dashed',color='r')
    ax2.hlines(1,0,1,linestyles='dashed',color='r')
    ax2.vlines(0,0,1,linestyles='dashed',color='r')
    ax2.vlines(1,0,1,linestyles='dashed',color='r')
    #Plot the background -> interlayer coupling
    long_X = np.linspace(-0.2,1.2,200)
    X,Y = np.meshgrid(long_X,long_X)
    l = 0.02       #length of arrow
    hw = 0.01       #arrow head width
    hl = 0.01       #arrow head length
    ax1.contourf(X,Y,fun_J(long_X,long_X),levels=20)
    ax2.contourf(X,Y,fun_J(long_X,long_X),levels=20)
    #Plot the arrows
    fac = grid//30     #plot 1 spin every "fac" of grid
    for i in range(grid//fac):
        x = (i*fac+fac//2)/grid
        for j in range(grid//fac):
            y = (j*fac+fac//2)/grid
            phi1 = phi_1[i*fac+fac//2,j*fac+fac//2]
            phi2 = phi_2[i*fac+fac//2,j*fac+fac//2]
            ax1.arrow(x - l/2*np.cos(phi1),y - l/2*np.sin(phi1),l*np.cos(phi1), l*np.sin(phi1),head_width=hw,head_length=hl,color='k')
            ax2.arrow(x - l/2*np.cos(phi2),y - l/2*np.sin(phi2),l*np.cos(phi2), l*np.sin(phi2),head_width=hw,head_length=hl,color='k')
    plt.show()

####################################################################################################################
def compute_interlayer(grid,A_M):
    """Computes the interlayer coupling as defined in Hejazi et al. 10.1073/pnas.2000347117

    Parameters
    ----------
    grid : int
        The number of points in each direction
    A_M : float
        The Moire lattice length

    Returns
    -------
    np.ndarray
        Interlayer coupling of shape (grid,grid).
    """
    #Reciprocal Moire vectors
    b_ = np.zeros((3,2))    #three arrays in x-y plane
    b_[0] = b1/A_M
    b_[1] = b2/A_M
    b_[2] = b_[0]-b_[1]
    G_M = np.linalg.norm(b_[0])
    #Moirè potential Phi
    Phi = np.zeros((grid,grid))
    latt = np.zeros((grid,grid,2))
    for i in range(grid):
        for j in range(grid):
            latt[i,j] = (i/grid*a1 + j/grid*a2)*A_M
            for a in range(3):
                Phi[i,j] += np.cos(np.dot(b_[a],latt[i,j]))
    return Phi

def name_Phi(grid,A_M,cluster=False):
    """Computes the filename of the interlayer coupling.

    Parameters
    ----------
    grid : int
        The number of points in each direction
    A_M : float
        The Moire lattice length
    cluster: bool, optional
        Wether we are in the cluster or not (default is False).

    Returns
    -------
    string
        The name of the .npy file containing the interlayer coupling.
        The directory name is NOT included.
    """
    return name_dir(cluster) + 'Phi_'+str(grid)+'_'+"{:.2f}".format(A_M)+'.npy'

def name_phi_as(alpha,beta,grid,A_M,cluster=False):
    """Computes the filenames of the symmetric and antisymmetric phases.

    Parameters
    ----------
    alpha: float
        Parameter alpha.
    beta: float
        Parameter beta.
    grid : int
        The number of points in each direction
    A_M : float
        The Moire lattice length
    cluster: bool, optional
        Wether we are in the cluster or not (default is False).

    Returns
    -------
    2-tuple
        Tuple of 2 elements containing the names of the .npy files.
        The directory name is NOT included.
    """
    return (name_dir(cluster)+'phi_s_'+"{:.4f}".format(alpha)+'_'+"{:.4f}".format(beta)+str(grid)+'_'+"{:.2f}".format(A_M)+'.npy',
            name_dir(cluster)+'phi_a_'+"{:.4f}".format(alpha)+'_'+"{:.4f}".format(beta)+str(grid)+'_'+"{:.2f}".format(A_M)+'.npy')

def name_dir(cluster=False):
    """Computes the directory name where to save the results.

    Parameters
    ----------
    cluster: bool, optional
        Wether we are in the cluster or not (default is False).

    Returns
    -------
    string
        The directory name.
    """
    dirname = '/home/users/r/rossid/CrBr3/results/' if cluster else '/home/dario/Desktop/git/CrBr3/results/'
    return dirname





























