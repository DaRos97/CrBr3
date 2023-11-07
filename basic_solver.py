import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS
import matplotlib.pyplot as plt
import functions as fs
import sys

cluster = False
alpha,beta = np.reshape(fs.values,(fs.pts_array**2,2))[int(sys.argv[1])]
#Check if already computed
dirname = '/home/users/r/rossid/CrBr3/results/' if cluster else '/home/dario/Desktop/git/CrBr3/results/'
filename_s = dirname + 'phi_s_' + "{:.4f}".format(alpha) + '_' + "{:.4f}".format(beta) + '.npy'
filename_a = dirname + 'phi_a_' + "{:.4f}".format(alpha) + '_' + "{:.4f}".format(beta) + '.npy'
try:
    phi_s = np.load(filename_s)
    phi_a = np.load(filename_a)
    print("Already computed")
    exit()
except:
    print("Computing...")

#Parameters of Moire lattice
A_M = 100 #related to theta somehow
grid = 100
#
b_ = np.zeros((3,2))    #three arrays in x-y plane
b_[0] = fs.b1/A_M
b_[1] = fs.b2/A_M
b_[2] = b_[0]-b_[1]
G_M = np.linalg.norm(b_[0])
#Moirè potential Phi
Phi = np.zeros((grid,grid))
latt = np.zeros((grid,grid,2))
for i in range(grid):
    for j in range(grid):
        latt[i,j] = (i/grid*fs.a1 + j/grid*fs.a2)*A_M
        for a in range(3):
            Phi[i,j] += np.cos(np.dot(b_[a],latt[i,j]))
#
args = (Phi,alpha,beta,A_M)
phi_s,phi_a = fs.compute_magnetization(args)

#Save
if cluster:
    np.save(filename_s,phi_s)
    np.save(filename_a,phi_a)

print("\nFinal energy: ",fs.compute_energy(phi_s,phi_a,args)/grid**2)

if 0:#input("Plot magnetization pattern?(y/N) ")=='y':   #plot
    #Interpolate Phi
    xpts,ypts = Phi.shape
    xpts = ypts = grid
    J = Phi
    XX = np.linspace(-2,2,4*xpts,endpoint=False)
    YY = np.linspace(-2,2,4*ypts,endpoint=False)
    big_J = np.zeros((4*xpts,4*xpts))
    big_J[:xpts,:xpts] = J; big_J[:xpts,xpts:2*xpts] = J; big_J[:xpts,2*xpts:3*xpts] = J; big_J[:xpts,3*xpts:] = J;
    big_J[xpts:2*xpts,:xpts] = J; big_J[xpts:2*xpts,xpts:2*xpts] = J; big_J[xpts:2*xpts,2*xpts:3*xpts] = J; big_J[xpts:2*xpts,3*xpts:] = J;
    big_J[2*xpts:3*xpts,:xpts] = J; big_J[2*xpts:3*xpts,xpts:2*xpts] = J; big_J[2*xpts:3*xpts,2*xpts:3*xpts] = J; big_J[2*xpts:3*xpts,3*xpts:] = J;
    big_J[3*xpts:,:xpts] = J; big_J[3*xpts:,xpts:2*xpts] = J; big_J[3*xpts:,2*xpts:3*xpts] = J; big_J[3*xpts:,3*xpts:] = J;
    fun_J = RBS(XX,YY,big_J)
    if 0:#input("Plot interlayer potential?(y/N)")=='y':   #plot interpolated moire interlayer function
        fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(20,20))
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        #J
        XX = np.linspace(0,1,xpts,endpoint=False)
        dx = XX[1]-XX[0]
        YY = np.linspace(0,1,ypts,endpoint=False)*np.sqrt(3)/2
        a,b = np.meshgrid(XX,YY)
        for i in range(xpts):
            a[i,:] -= 1/2*i*dx
        ax1_p = ax1.contourf(a,b,J)
        ax1.set_title("Defined coupling")
        fig.colorbar(ax1_p)
        #Interpolation
        xpts = ypts = 200
        XX = np.linspace(0,1,xpts,endpoint=False)
        dx = XX[1]-XX[0]
        YY = np.linspace(0,1,ypts,endpoint=False)*np.sqrt(3)/2
        a,b = np.meshgrid(XX,YY)
        for i in range(xpts):
            a[i,:] -= 1/2*i*dx
        ax2_p = ax2.contourf(a,b,fun_J(XX,XX))
        ax2.set_title("Interpolated interlayer coupling")
        fig.colorbar(ax2_p)
        plt.show()
        exit()

    #Actual plot
    fac = grid//30     #plot 1 spin every "fac" of grid
    fs.plot_magnetization(phi_s,phi_a,fun_J,fac)



































