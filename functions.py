import numpy as np
import matplotlib.pyplot as plt 

a1 = np.array([1,0])
a2 = np.array([-1/2,np.sqrt(3)/2])

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

def der_2(phi):
    #Compute d^2(phi)/dx^2 + d^2(phi)/dy^2
    dx = 1/phi.shape[0]
    dy = 1/phi.shape[1]      ##### dy=?????
    return ( (np.roll(phi,2,axis=0)-2*np.roll(phi,1,axis=0)+np.roll(phi,0,axis=0))/dx**2 + 
             (np.roll(phi,-2,axis=1)-2*np.roll(phi,-1,axis=1)+np.roll(phi,0,axis=1))/dy**2
             )
    
def grad_H(phi_s,phi_a,tt,args):
    Phi,alpha,beta,G_M = args
    if tt=='s':
        return beta*np.sin(phi_s)*np.cos(phi_a) - der_2(phi_s)
    elif tt=='a':
        return (beta*np.cos(phi_s)+alpha*Phi)*np.sin(phi_a) - der_2(phi_a)

def initial_point(args):
    Phi,alpha,beta,G_M = args
    grid = Phi.shape[0]
    initial_ansatz = ['twisted','constant']
    sol=initial_ansatz[1]
    if sol=='twisted':      #ansatz for alpha,beta<<1
        delta = beta/alpha**2
        print("Using collinear solution as starting point, with delta= ",delta)
        if abs(delta)>3/2:
            print("delta too large for this approximation")
            exit()
        phi0 = np.arccos(-2/3*delta)
        const = 1/2-np.tan(phi0)**(-2)
        temp = np.linspace(0,1,grid)
        phi_s = np.ones((grid,grid))*np.pi
        phi_a = phi0 - alpha*np.sin(phi0)*(Phi-const)
    elif sol=='constant':
        ss = np.pi
        aa = np.pi*0.5
        print("Using constant solution as starting point, with values phi_s= ",ss,", phi_a= ",aa)
        phi_s = np.ones((grid,grid))*ss
        phi_a = np.ones((grid,grid))*aa
    return phi_s, phi_a

def exit_loop_condition(d_0,d_1):
    if d_1<1e-3 or abs(d_0-d_1)<1e-3:
        return True
    else:
        return False

def compute_energy2(phi_i,*args):
    Phi,alpha,beta,G_M = args
    grid = Phi.shape[0]
    return compute_energy(np.reshape(phi_i[:grid**2],(grid,grid)),np.reshape(phi_i[grid**2:],(grid,grid)),args)

def compute_energy(phi_s,phi_a,args):
    Phi,alpha,beta,G_M = args
    grid = phi_s.shape[0]
    dx = 1/grid
    dy = 1/grid
    grad_2s = np.absolute((np.roll(phi_s,1,axis=0)-phi_s)/dx)**2 + np.absolute((np.roll(phi_s,-1,axis=1)-phi_s)/dy)**2
    grad_2a = np.absolute((np.roll(phi_a,1,axis=0)-phi_a)/dx)**2 + np.absolute((np.roll(phi_a,-1,axis=1)-phi_a)/dy)**2
    energy = 1/2*(grad_2s+grad_2a)-np.cos(phi_a)*(alpha*Phi+beta*np.cos(phi_s))
    H = energy.sum()/grid**2
    return H
def grad_e(phi_s,phi_a,tt,args):
    dd = 0.01
    dphi = np.ones(phi_s.shape)*dd
    if tt == 's':
        Ep = compute_energy(phi_s+dphi,phi_a,args)
    if tt == 'a':
        Ep = compute_energy(phi_s,phi_a+dphi,args)
    return (Ep-compute_energy(phi_s,phi_a,args))/dd



def compute_magnetization(rho,d,Jp,Phi,G_M):
    alpha = 2*Jp/rho/G_M**2
    beta = 2*d/rho/G_M**2
    alpha = 1
    beta = 0.
    print("Parameters: alpha="+"{:.4f}".format(alpha)+", beta="+"{:.4f}".format(beta))
    #Start by defining the energy as a function of phi_s and phi_a
    #Define initial value of phi_s and phi_a
    learn_rate = 0.5        #to fix for finding a global minimum
    args = (Phi,alpha,beta,G_M)
    phi_s,phi_a = initial_point(args)
    diff_0 = 1e9
    diff_1 = 1e8
    if 1:   #minimization
        grid = Phi.shape[0]
        phi_i = np.zeros(2*grid**2)
        phi_i[:grid**2] = np.reshape(phi_s,(grid**2))
        phi_i[grid**2:] = np.reshape(phi_a,(grid**2))
        from scipy.optimize import minimize
        res = minimize(compute_energy2, 
                x0 = phi_i,
                args = args
                )
        phi_s = np.reshape(res.x[:grid**2],(grid,grid))
        phi_a = np.reshape(res.x[grid**2:],(grid,grid))
        print("Finished...? ",res.nit," iterations")
        print("Energy: ",res.fun)
    elif 0:
        phi_new_s = phi_s
        phi_new_a = phi_a
        print("Starting minimization")
        step = 1
        while True:
            print("Step ",step)
            dHs = grad_H(phi_s,phi_a,'s',args)
            dHa = grad_H(phi_s,phi_a,'a',args)
            dEs = grad_e(phi_s,phi_a,'s',args)
            dEa = grad_e(phi_s,phi_a,'a',args)
            #
            phi_new_s = phi_s + learn_rate*dEs
            phi_new_a = phi_a + learn_rate*dEa
            #
            E = compute_energy(phi_new_s,phi_new_a,args)
            diff_0 = diff_1
            diff_1 = (np.absolute(phi_s-phi_new_s) + np.absolute(phi_a-phi_new_a)).sum()
            if exit_loop_condition(diff_0,diff_1):
                print("Minimization criterion reached!")
#                print("phi_s: ",phi_s)
#                print("phi_a: ",phi_a)
                print("energy = ",E)
                break
            else:
                print("Difference: ",diff_1)
                print("energy = ",E)
                phi_s = phi_new_s
                phi_a = phi_new_a
            #
            if 0:
                input("Going to next step")
            step += 1

    else:
        phi_new_s = phi_s
        phi_new_a = phi_a
    return phi_s,phi_a

def plot_magnetization(phi_s,phi_a,fun_J,fac):
    grid = phi_s.shape[0]
    phi_1 = (phi_s+phi_a)/2
    phi_2 = (phi_s-phi_a)/2

    #Plot magnetization patterns
    #Two different plots
    fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(20,20))
    ax1.hlines(0,0,1,linestyles='dashed',color='r')
    ax1.hlines(1,0,1,linestyles='dashed',color='r')
    ax1.vlines(0,0,1,linestyles='dashed',color='r')
    ax1.vlines(1,0,1,linestyles='dashed',color='r')
    ax2.hlines(0,0,1,linestyles='dashed',color='r')
    ax2.hlines(1,0,1,linestyles='dashed',color='r')
    ax2.vlines(0,0,1,linestyles='dashed',color='r')
    ax2.vlines(1,0,1,linestyles='dashed',color='r')
    long_X = np.linspace(-0.2,1.2,200)
    X,Y = np.meshgrid(long_X,long_X)
    l = 0.02       #length of arrow
    hw = 0.01
    hl = 0.01
    ax1.contourf(X,Y,fun_J(long_X,long_X),levels=20)
    ax2.contourf(X,Y,fun_J(long_X,long_X),levels=20)
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


def grad_phi(phi):
    grid = phi.shape[0]
    dx = 1/A_M
    dy = 1/A_M
    der_x = np.zeros((grid,grid))
    for j in range(grid):
        for i in range(grid-1):
            der_x[i,j] = (phi[i+1,j]-phi[i,j])/dx
        der_x[-1,j] = (phi[0,j]-phi[-1,j])/dx
    der_y = np.zeros((grid,grid))
    for i in range(grid):
        for j in range(grid-1):
            der_x[i,j] = (phi[i,j+1]-phi[i,j])/dy
        der_x[i,-1] = (phi[i,0]-phi[i,-1])/dy
    return (der_x,der_y)


































