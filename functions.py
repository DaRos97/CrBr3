import numpy as np

a1 = np.array([1,0])
a2 = np.array([-1/2,np.sqrt(3)/2])

def find_closest(lattice,A,theta,site,UC_):
    #Finds the closest lattice site to the coordinates "site". The lattice is stored in "lattice", has
    #a lattice length "A" and the search can be constrained to the unit cell "UC_".

    #Lattice has shape nx,ny,2->unit cell index,2->x and y.
    X,Y,W,Z = lattice.shape
    #Smallest y-difference in A and B sublattice
    miny_A = np.min(abs(np.ones(Y)*site[1]-np.arange(Y)*A*np.sqrt(3)/2))
    miny_B = np.min(abs(np.ones(Y)*site[1]-np.arange(Y)*A*np.sqrt(3)/2-A/np.sqrt(3)*np.ones(Y)))
    if UC_=='nan':
        if miny_A < miny_B:
            UC = 0
            argy = np.argmin(abs(np.ones(Y)*site[1]-np.arange(Y)*A*np.sqrt(3)/2))
        else:
            UC = 1
            argy = np.argmin(abs(np.ones(Y)*site[1]-np.arange(Y)*A*np.sqrt(3)/2-A/np.sqrt(3)*np.ones(Y)))
    elif UC_==0:
        argy = np.argmin(abs(np.ones(Y)*site[1]-np.arange(Y)*A*np.sqrt(3)/2))
        UC = UC_
    elif UC_==1:
        argy = np.argmin(abs(np.ones(Y)*site[1]-np.arange(Y)*A*np.sqrt(3)/2-A/np.sqrt(3)*np.ones(Y)))
        UC = UC_
    #Smalles x difference
    argx = np.argmin(abs(np.ones(X)*site[0]-np.arange(X)*A+np.arange(Y)[argy]/2*A))
    return argx,argy,UC

def compute_lattices(A_1,A_2,theta,xxx,yyy):
    l1 = np.zeros((xxx,yyy,2,2))
    l2 = np.zeros((xxx,yyy,2,2))
    for i in range(xxx):
        for j in range(yyy):
            cord = i*a1 + j*a2
            l1[i,j,0] = A_1*cord
            l1[i,j,1] = l1[i,j,0] + np.array([0,A_1/np.sqrt(3)])
            l1[i,j,0] = np.tensordot(R_z(theta/2),l1[i,j,0],1)
            l1[i,j,1] = np.tensordot(R_z(theta/2),l1[i,j,1],1)
            l2[i,j,0] = A_2*cord
            l2[i,j,1] = l2[i,j,0] + np.array([0,A_2/np.sqrt(3)])
            l2[i,j,0] = np.tensordot(R_z(-theta/2),l2[i,j,0],1)
            l2[i,j,1] = np.tensordot(R_z(-theta/2),l2[i,j,1],1)
    return l1,l2

def moire_length(A_1,A_2,theta):
    return 1/np.sqrt(1/A_1**2+1/A_2**2-2*np.cos(theta)/(A_1*A_2))

def R_z(t):
    R = np.zeros((2,2))
    R[0,0] = np.cos(t)
    R[0,1] = -np.sin(t)
    R[1,0] = np.sin(t)
    R[1,1] = np.cos(t)
    return R

def grad_phi(phi,A_M):
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

def energy(phi_s,phi_a,Phi,alpha,beta,A_M):
    grid = phi_s.shape[0]
    grad_phi_s = grad_phi(phi_s,A_M)
    grad_s_2 = np.zeros((grid,grid))
    grad_phi_a = grad_phi(phi_a,A_M)
    grad_a_2 = np.zeros((grid,grid))
    cos_s = np.zeros((grid,grid))
    cos_a = np.zeros((grid,grid))
    energy = np.zeros((grid,grid))
    for i in range(grid):
        for j in range(grid):
            grad_s_2 = np.sqrt(grad_phi_s[0][i,j]**2+grad_phi_s[1][i,j]**2)
            grad_a_2 = np.sqrt(grad_phi_a[0][i,j]**2+grad_phi_a[1][i,j]**2)
            cos_s = np.cos(phi_s[i,j])
            cos_a = np.cos(phi_a[i,j])
            energy[i,j] = 1/2*(grad_s_2+grad_a_2) - (alpha*Phi[i,j]+beta*cos_s)*cos_a

    H = energy.sum()/grid**2
    return H
    










































