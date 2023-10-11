import numpy as np


def find_closest(lattice,A,site,UC_):
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
        argy = np.argmin(abs(np.ones(Y)*site[1]-np.arange(Y)*A*np.sqrt(3)/2))
        UC = UC_
    #Smalles x difference
    argx = np.argmin(abs(np.ones(X)*site[0]-np.arange(X)*A+np.arange(Y)[argy]/2*A))
    return argx,argy,UC

def find_interlayer(data,S1,S2):
    pts,sets = data.shape
    index = np.argmin(abs(np.ones(pts)*S1-data[:,0])+abs(np.ones(pts)*S2-data[:,1]))
    return data[index,2]-data[index,3], index



