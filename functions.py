import numpy as np
import inputs
from scipy.interpolate import RectBivariateSpline as RBS
from pathlib import Path

#Triangular lattice
a1 = np.array([1,0])
a2 = np.array([-1/2,np.sqrt(3)/2])
d = np.array([0,1/np.sqrt(3)])  #vector connecting the two sublattices
b1 = np.array([1,1/np.sqrt(3)])*2*np.pi
b2 = np.array([0,2/np.sqrt(3)])*2*np.pi
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
    if inputs.moire_type=='general':
        e_xx = inputs.moire_pars['general']['e_xx']
        e_yy = inputs.moire_pars['general']['e_yy']
        e_xy = inputs.moire_pars['general']['e_xy']
        strain_tensor = np.array([[e_xx,e_xy],[e_xy,e_yy]])
    elif inputs.moire_type=='uniaxial':
        eps = inputs.moire_pars['uniaxial']['eps']
        ni = inputs.moire_pars['uniaxial']['ni']
        phi = inputs.moire_pars['uniaxial']['phi']
        strain_tensor = np.matmul(np.matmul(R_z(-phi).T,np.array([[eps,0],[0,-ni*eps]])),R_z(-phi))
    elif inputs.moire_type=='biaxial':
        eps = inputs.moire_pars['biaxial']['eps']
        strain_tensor = np.identity(2)*eps
    elif inputs.moire_type=='shear':
        e_xy = inputs.moire_pars['shear']['e_xy']
        phi = inputs.moire_pars['shear']['phi']
        strain_tensor = np.matmul(np.matmul(R_z(-phi).T,np.array([[0,e_xy],[e_xy,0]])),R_z(-phi))
    #
    theta = inputs.moire_pars['theta']
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
    n_x = n_y = 8       #number of moirè lenths to include in l1,l2
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
    pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0 = inputs.args_general
    rg = pts_per_fit
    smooth_phi = np.zeros((grid,grid))
    for i in range(-rg,rg+1):
        for j in range(-rg,rg+1):
            smooth_phi += np.roll(np.roll(phi,i,axis=0),j,axis=1)
    smooth_phi /= (1+2*rg)**2
    return smooth_phi

def get_dft_data(cluster='loc'):
    data_fn = inputs.home_dirname[cluster]+"Data/CrBr3_interlayer.npy"
    if Path(data_fn).is_file():
        return np.load(data_fn)
    #Compute it
    data_marco_fn = inputs.home_dirname[cluster]+"Data/CrBr3_scan.txt"
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

def plot_Phi(Phi,title=''):
    """Plot interlayer.

    Parameters
    ----------
    Phi : ndarray
        Interlayer poptential.
    title : string, optional (default = '').
        Plot title.
    """
    grid = Phi.shape[0]
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex": True,})
    s_ = 20
#    Phi = symmetrize(Phi)
    XX = np.linspace(-1,2,3*grid,endpoint=False)
    big_Phi = extend(Phi)
    fun_Phi = RBS(XX,XX,big_Phi)
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
    fig, ax = plt.subplots(figsize=(10,10))
    ax.axis('off')
    ax.set_aspect(1.)
    ax.contour(X,Y,fun_Phi(long_X,long_Y),levels=[0,],colors=('r',),linestyles=('-',),linewidths=(0.5))
    surf = ax.contourf(X,Y,fun_Phi(long_X,long_Y),levels=20)
    ax.vlines(1/2,-1/2/s3,1/2/s3,linestyles='dashed',color='r')
    ax.vlines(-1/2,-1/2/s3,1/2/s3,linestyles='dashed',color='r')
    for i,x in enumerate([xx14,xx23]):
        for j in range(2):
            ax.plot(x,line(x,*pars[i][j]),linestyle='dashed',color='r')
    ax.set_xlim(-0.6,0.6)
    ax.set_ylim(-0.65,0.65)
    #ax.set_title("Interlayer potential of strained Moirè lattice",size=s_)
#    ax.set_title("interlayer potential",size=s_)
    plt.show()

def extend(phi,nn=3):
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
    grid = phi.shape[0]
    L = np.zeros((nn*grid,nn*grid))
    for i in range(nn):
        for j in range(nn):
            L[i*grid:(i+1)*grid,j*grid:(j+1)*grid] = phi
    return L

def name_dir_Phi(cluster='loc'):
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
    dirname = inputs.home_dirname[cluster]+'Phi_values/'
    return dirname

def name_Phi(cluster='loc'):
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
    pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0 = inputs.args_general
    return name_dir_Phi(cluster) + 'Phi_'+general_fn([grid,],'.npy')

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
