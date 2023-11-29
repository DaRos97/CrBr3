import numpy as np
import random
import inputs
import os
from scipy.interpolate import RectBivariateSpline as RBS
from pathlib import Path

#Triangular lattice
a1 = np.array([1,0])
a2 = np.array([-1/2,np.sqrt(3)/2])
b1 = np.array([1,1/np.sqrt(3)])*2*np.pi
b2 = np.array([0,2/np.sqrt(3)])*2*np.pi
#
def compute_magnetization(Phi,pars,args_minimization):
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
    args_minimization : dic
        'rand_m' -> int, number of random initial seeds,
        'maxiter' -> int, max number of update evaluations.
        'disp' -> bool, diplay messages.

    Returns
    -------
    tuple
        Symmetric and antisymmetric phases at each position (grid,grid) of the Moirè unit cell.
    """
    gamma,alpha,beta = pars
    pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = inputs.args_general
    #Variables for storing best solution
    min_E = 1e10
    result = np.zeros((2,grid,grid))
    for sss in range(args_minimization['rand_m']):
        if min_E<1e8 and not args_minimization['disp']:
            np.save(name_phi(pars,True),result)
        E = []  #list of energies for the while loop
        diff_H = [1e20]  #list of dH for the while loop
        if args_minimization['disp']:
            print("Starting minimization step ",str(sss))
        #Initial condition  -> Improve 
        fs = 0.5 if sss==1 else random.random()
        fa = 0.5 if sss==1 else random.random()
        ans = 0 if sss==0 else 1            #Use twisted-s ansatz for first evaluation or constant phi_s/a=pi
        #Compute first state and energy
        phi_s,phi_a = initial_point(Phi,pars,fs,fa,ans)
        d_phi = (compute_derivatives(phi_s,1),compute_derivatives(phi_a,1))
        E.append(compute_energy(phi_s,phi_a,Phi,pars,d_phi))
        #Initiate learning rate and minimization loop
        step = 1        #initial step
        while True:
            learn_rate = learn_rate_0*random.random()
            #Energy gradients
            dHs = grad_H(phi_s,phi_a,'s',Phi,pars,compute_derivatives(phi_s,2))
            dHa = grad_H(phi_s,phi_a,'a',Phi,pars,compute_derivatives(phi_a,2))
            #Update phi
            phi_s += learn_rate*dHs
            phi_a += learn_rate*dHa
            #New energy
            d_phi = (compute_derivatives(phi_s,1),compute_derivatives(phi_a,1))
            E.insert(0,compute_energy(phi_s,phi_a,Phi,pars,d_phi))
            #Check if dHs and dHa are very small
            diff_H.insert(0,np.sum(np.absolute(dHs)+np.absolute(dHa)))
            if args_minimization['disp']:
                print("energy step ",step," is ",E[1]," ,dH at ",diff_H[0])
            #Exit checks
            if check_energies(E):   #stable energy
                if E[0]<min_E:
                    min_E = E[0]
                    result[0] = np.copy(phi_s)
                    result[1] = np.copy(phi_a)
                break
            if diff_H[0]>diff_H[1] or E[0]>E[1]:    #worse solution
                phi_s -= learn_rate*dHs
                phi_a -= learn_rate*dHa
                del E[0]
                del diff_H[0]
                if E[0]<min_E:
                    min_E = E[0]
                    result[0] = np.copy(phi_s)
                    result[1] = np.copy(phi_a)
                break
            if E[0] > 1e10: #VERY bad situation
                print("bullshit")
                break
            #Max number of steps scenario
            if step > args_minimization['maxiter']:
                if sss == 0:    #If this happens for the first minimization step, save a clearly fake one for later comparison
                    min_E = 1e8
                    result[0] = np.ones((grid,grid))*20
                    result[1] = np.ones((grid,grid))*20
                break
            step += 1
            #
        if args_minimization['disp']:
            print("Minimum energy at ",E[0]," ,dH at ",diff_H[0])
            plot_phis(result[0],result[1],'phi_s and phi_a')
            plot_magnetization(result[0],result[1],Phi,pars)
#            input()
    return result

def initial_point(Phi,pars,fs,fa,ans):
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
    gamma,alpha,beta = pars
    pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = inputs.args_general
    initial_ansatz = ['twisted-s','constant']
    sol = initial_ansatz[ans]
    if sol=='twisted-s':      #ansatz for alpha,beta<<1
        if alpha > 0:
            delta = beta/alpha**2
        else:
            delta = 10
        if abs(delta)>3/2:
            print("delta ",str(delta)," too large for twisted-s, switching to constant initial condition at pi,pi")
            return initial_point(Phi,pars,0,0,1)
        phi0 = np.arccos(2/3*delta)
        const = 1/2-np.tan(phi0)**(-2)
        phi_s = np.ones((grid,grid))*np.pi
        phi_a = phi0 - alpha*np.sin(phi0)*(Phi-const)
    elif sol=='constant':
        phi_s = np.ones((grid,grid))*2*np.pi*fs
        phi_a = np.ones((grid,grid))*2*np.pi*fa
    return phi_s, phi_a

def compute_energy(phi_s,phi_a,Phi,pars,d_phi):
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

    Returns
    -------
    float
        Energy density summed over all sites.
    """
    gamma,alpha,beta = pars
    pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = inputs.args_general
    dx = dy = A_M/grid
    #Old derivative squared
    grad_2s = np.absolute(d_phi[0][0])**2 + np.absolute(d_phi[0][1])**2
    grad_2a = np.absolute(d_phi[1][0])**2 + np.absolute(d_phi[1][1])**2
    kin_part = 1/2*(grad_2s+grad_2a)
    energy = kin_part - np.cos(phi_a)*(alpha*Phi+beta*np.cos(phi_s)) - 2*gamma*np.cos(phi_s/2)*np.cos(phi_a/2)
    H = energy.sum()/grid**2
    return H

def grad_H(phi_s,phi_a,tt,Phi,pars,d2_phi):
    """Computes evolution step dH/d phi.

    Parameters
    ----------
    phi_s: np.ndarray
        Symmetric phases.
    phi_a: np.ndarray
        A-symmetric phases.
    tt : char
        Determines which functional derivative to compute (wrt phi_s or phi_a).
    Phi : np.ndarray
        Interlayer coupling of size (grid,grid)
    alpha: float
        Parameter alpha.
    beta: float
        Parameter beta.

    Returns
    -------
    np.ndarray
        Gradient of Hamiltonian on the (grid,grid) space.
    """
    gamma,alpha,beta = pars
    pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = inputs.args_general
    res = -d2_phi[0]-d2_phi[1]
    if tt=='s':
        return res + beta*np.sin(phi_s)*np.cos(phi_a) + gamma*np.cos(phi_a/2)*np.sin(phi_s/2)
    elif tt=='a':
        return res + (beta*np.cos(phi_s)+alpha*Phi)*np.sin(phi_a) + gamma*np.cos(phi_s/2)*np.sin(phi_a/2)

def smooth(phi):
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
    pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = inputs.args_general
    rg = pts_per_fit
    smooth_phi = np.zeros((grid,grid))
    for i in range(-rg,rg+1):
        for j in range(-rg,rg+1):
            smooth_phi += np.roll(np.roll(phi,i,axis=0),j,axis=1)
    smooth_phi /= (1+2*rg)**2
    xx = np.linspace(0,A_M,grid,endpoint=False)
    fun = RBS(xx,xx,smooth_phi)
    return smooth_phi, fun

def empty_fun(x):
    """Empty function.

    Parameters
    ----------
    x: np.ndarray
        Function on a grid.

    Returns
    -------
    np.ndarray, function
        Retruns x and the function interpolating it.
    """
    pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = inputs.args_general
    return x,smooth(x)[1]

def compute_derivatives(phi,n):
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
    pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = inputs.args_general
    qm = 4*np.pi/np.sqrt(3)/A_M
    xx = np.linspace(0,A_M,grid,endpoint=False)
    #
    smooth_or_not = smooth if n == 2 else empty_fun
    #Interpolate phase
    fun = smooth_or_not(phi)[1]
    #derivatives
    dn_phi_x = smooth_or_not(fun.partial_derivative(n,0)(xx,xx)/qm**n)[0]
    dn_phi_y = smooth_or_not(fun.partial_derivative(0,n)(xx,xx)/qm**n)[0]
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
    lim = 1e-8
    n_check = nn if len(list_E)>nn else len(list_E)-1
    for i in range(n_check):
        if abs(list_E[i]-list_E[i+1]) > lim:
            return False
    return True

def plot_phis(phi_1,phi_2,txt_title='mah'):
    """Plot the phases phi_1 and phi_2 in a 3D graph

    Parameters
    ----------
    phi_1 : np.ndarray
        Phases defined on (grid,grid) space.
    phi_2 : np.ndarray
        Phases defined on (grid,grid) space.
    txt_title : string (optional)
        Title of the plot.

    """
    pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = inputs.args_general
    import matplotlib.pyplot as plt 
    from matplotlib import cm
    #
    fig, (ax1,ax2) = plt.subplots(1,2,subplot_kw={"projection": "3d"},figsize=(20,10))
    plt.suptitle(txt_title)
    X,Y = np.meshgrid(np.linspace(0,1,grid,endpoint=False),np.linspace(0,1,grid,endpoint=False))
    surf = ax1.plot_surface(X, Y, phi_1, cmap=cm.coolwarm,
               linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    surf = ax2.plot_surface(X, Y, phi_2, cmap=cm.coolwarm,
               linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def extend(phi):
    """Extend the domain of phi from 0,A_M to -A_M,2*A_M by copying it periodically.

    Parameters
    ----------
    phi: np.ndarray
        Function on a grid to extend.

    Returns
    -------
    np.ndarray
        Values of phi periodically repeated on larger domain.
    """
    grid = phi.shape[0]
    L = np.zeros((3*grid,3*grid))
    for i in range(3):
        for j in range(3):
            L[i*grid:(i+1)*grid,j*grid:(j+1)*grid] = phi
    return L

def plot_magnetization(phi_s,phi_a,Phi,pars):
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

    """
    gamma,alpha,beta = pars
    pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = inputs.args_general
    import matplotlib.pyplot as plt 
    #Interpolate Phi
    XX = np.linspace(-1,2,3*grid,endpoint=False)
    big_Phi = extend(Phi)
    fun_Phi = RBS(XX,XX,big_Phi)
    #Single layer phases
    phi_1 = (phi_s+phi_a)/2
    phi_2 = (phi_s-phi_a)/2
    #Background -> interlayer coupling
    long_X = np.linspace(-1,1,2*grid,endpoint=False)
    long_Y = np.linspace(-1,1,2*grid,endpoint=False)
    X,Y = np.meshgrid(long_X,long_Y)
    X = X-Y/2
    Y = Y/2*np.sqrt(3)
    #Box the Moirè unit cell
    s3 = np.sqrt(3)
    def line(x,y0,q):
        return y0+q*x
    xx14 = np.linspace(-1/2,0,100)
    xx23 = np.linspace(0,1/2,100)
    pars = [[(1/s3,1/s3),(-1/s3,-1/s3)],
            [(1/s3,-1/s3),(-1/s3,1/s3)]]
    #Plot the arrows
    l = 0.02       #length of arrow
    hw = 0.01       #arrow head width
    hl = 0.01       #arrow head length
    fac = grid//20     #plot 1 spin every "fac" of grid
    def inside_UC(a,b):
        if a > 1/2:
            return inside_UC(a-1,b)
        elif b>1/s3+a/s3:
            return inside_UC(a+1/2,b-s3/2)
        elif b>1/s3-a/s3:
            return inside_UC(a-1/2,b-s3/2)
        else:
            return a,b
    phi_ = [phi_1,phi_2]
    #Plot magnetization patterns -> two different plots
    fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(20,20))
    for ind,ax in enumerate([ax1,ax2]):
        ax.axis('off')
        ax.set_aspect(1.)
        surf = ax.contourf(X,Y,fun_Phi(long_X,long_Y),levels=20)
        ax.vlines(1/2,-1/2/s3,1/2/s3,linestyles='dashed',color='r')
        ax.vlines(-1/2,-1/2/s3,1/2/s3,linestyles='dashed',color='r')
        for i,x in enumerate([xx14,xx23]):
            for j in range(2):
                ax.plot(x,line(x,*pars[i][j]),linestyle='dashed',color='r')
        for i in range(grid//fac):
            for j in range(grid//fac):
                x_c = (i*fac)/grid
                y_c = (j*fac)/grid
                x = x_c-y_c/2
                y = y_c/2*np.sqrt(3)
                x,y = inside_UC(x,y)
                phi = phi_[ind][i*fac,j*fac]
                ax.arrow(x - l/2*np.sin(phi),y - l/2*np.cos(phi),l*np.sin(phi), l*np.cos(phi),head_width=hw,head_length=hl,color='k')
        ax.set_xlim(-0.6,0.6)
        ax.set_ylim(-0.65,0.65)
    plt.suptitle("alpha/(1+alpha) = "+"{:.4f}".format(alpha/(1+alpha))+",  beta/(1+beta) = "+"{:.4f}".format(beta/(1+beta))+", gamma = "+"{:.4f}".format(gamma))
    plt.show()

####################################################################################################################
def compute_interlayer():
    """Computes the interlayer coupling as defined in Hejazi et al. 10.1073/pnas.2000347117

    Returns
    -------
    np.ndarray
        Interlayer coupling.
    """
    pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = inputs.args_general
    #Reciprocal Moire vectors
    b_ = np.zeros((3,2))    #three arrays in x-y plane
    b_[0] = b1/A_M
    b_[1] = b2/A_M
    b_[2] = b_[0]-b_[1]
    G_M = np.linalg.norm(b_[0])
    #Moirè potential Phi
    Phi = np.zeros((grid,grid))
    for i in range(grid):
        for j in range(grid):
            x = (i/grid*a1 + j/grid*a2)*A_M
            for a in range(3):
                Phi[i,j] += np.cos(np.dot(b_[a],x))
    return Phi

def name_dir_Phi(cluster=False):
    """Computes the directory name where to save the interlayer potential.

    Parameters
    ----------
    cluster: bool, optional
        Wether we are in the cluster or not (default is False).

    Returns
    -------
    string
        The directory name.
    """
    dirname = '/home/users/r/rossid/CrBr3/Phi_values/' if cluster else '/home/dario/Desktop/git/CrBr3/Phi_values/'
    return dirname

def name_Phi(cluster=False):
    """Computes the filename of the interlayer coupling.

    Parameters
    ----------
    cluster: bool, optional
        Wether we are in the cluster or not (default is False).

    Returns
    -------
    string
        The name of the .npy file containing the interlayer coupling.
    """
    pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = inputs.args_general
    return name_dir_Phi(cluster) + 'Phi_'+str(grid)+'_'+"{:.2f}".format(A_M)+'.npy'

def name_dir_phi(cluster=False):
    """Computes the directory name where to save the results.

    Parameters
    ----------
    cluster: bool, optional
        Wether we are in the cluster or not (default is False).

    Returns
    -------
    string
        The directory name of the minimization result.
    """
    pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = inputs.args_general
    name_particular = 'results_'+str(grid)+'_'+str(pts_per_fit)+'_'+"{:.4f}".format(learn_rate_0)+'_'+"{:.2f}".format(A_M)+'/'
    dirname = '/home/users/r/rossid/CrBr3/results/' if cluster else '/home/dario/Desktop/git/CrBr3/results/'
    return dirname+name_particular

def name_phi(pars,cluster=False):
    """Computes the filename of the result of the minimization.

    Parameters
    ----------
    alpha: float
        Parameter alpha.
    beta: float
        Parameter beta.
    cluster: bool, optional
        Wether we are in the cluster or not (default is False).

    Returns
    -------
    string
        Filename of the .npy file.
    """
    gamma,alpha,beta = pars
    return name_dir_phi(cluster)+'phi_'+"{:.4f}".format(alpha)+'_'+"{:.4f}".format(beta)+'_'+"{:.4f}".format(gamma)+'.npy'

def compute_parameters():
    """Computes the grid of alpha/beta points to consider in order to build up the phase diagram.
    The phase diagram is computed in scale of a/(1+a), so we consider pts_array values 'g' between 
    0 and 1 and compute alpha/beta as alpha(beta)=1/(1-g).

    Returns
    -------
    np.ndarray
        Matrix of values of size (pts_array,pts_array,2).
    """
    pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = inputs.args_general
    values = np.zeros((pts_gamma,pts_array,pts_array,3))
    ab_array = np.linspace(0,1,pts_array,endpoint=False)
    g_array = np.linspace(0,1,pts_gamma,endpoint=False)
    for i in range(pts_gamma):
        for j in range(pts_array):
            for k in range(pts_array):
                values[i,j,k,0] = g_array[i]
                values[i,j,k,1] = ab_array[j]/(1-ab_array[j])
                values[i,j,k,2] = ab_array[k]/(1-ab_array[k])
    pars_arr = np.reshape(values,(pts_array*pts_array*pts_gamma,3))
    return pars_arr

def check_directories(cluster):
    """Check if the directories where to store the results exist and if not creates them.

    Parameters
    ----------
    cluster: bool, optional
        Wether we are in the cluster or not (default is False).

    """
    if not Path(name_dir_phi(cluster)).is_dir():
        os.system('mkdir '+name_dir_phi(cluster))
    if not Path(name_dir_Phi(cluster)).is_dir():
        os.system('mkdir '+name_dir_Phi(cluster))



##############################################################################
##############################################################################
##############################################################################

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





##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
#Other functions

def other_compute_derivatives(phi,n):
    diff = qm*A_M/grid
    order = '2'
    coeff = inputs.coeff_der[str(n)][order]
    sm_phi,fun = smooth(phi) if n==2 else (phi,0)
    dn_phi_x = np.zeros((grid,grid))
    dn_phi_y = np.zeros((grid,grid))
    for i in range(len(coeff)):
        dn_phi_x += coeff[i]*np.roll(sm_phi,-i,axis=0)/diff**n
        dn_phi_y += coeff[i]*np.roll(sm_phi,-i,axis=1)/diff**n
    if 0 and n ==2:
        xx = np.linspace(0,A_M,grid,endpoint=False)
        plot_phis(dn_phi_x,smooth(fun.partial_derivative(n,0)(xx,xx)/qm**n)[0],str(n))
    dn_phi = (smooth(dn_phi_x)[0],smooth(dn_phi_y)[0]) if n==2 else (dn_phi_x,dn_phi_y)
    return dn_phi

def other_smooth(phi):
    pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = inputs.args_general
    #Extend of factor 3
    xx_ext = np.linspace(-A_M,2*A_M,3*grid,endpoint=False)
    phi_ext = extend(phi)
    #Interpolate on less points
    pts = 3*grid // pts_per_fit        #21 points per axis per unit cell
    init = 0
    xx_less = xx_ext[init::pts]
    fun = RBS(xx_less,xx_less,phi_ext[init::pts,init::pts],kx=5,ky=5)
    #Compute on original grid
    xx = np.linspace(0,A_M,grid,endpoint=False)
    phi_new = fun(xx,xx)
    return phi_new, fun


















