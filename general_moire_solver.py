import numpy as np
import functions as fs
import os
from scipy.interpolate import RectBivariateSpline as RBS
import matplotlib.pyplot as plt
import sys,getopt

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "M:", ["A1=","A2=","theta=","pts="])
    A_1 = 1.0
    A_2 = 1.0
    theta = 0
    xpts = ypts = 100
except:
    print("Error in passed parameters of "+sys.argv[0])
    exit()
for opt, arg in opts:
    if opt == '--A1':
        A_1 = float(arg)
    if opt == '--A2':
        A_2 = float(arg)
    if opt == '--theta':
        theta = float(arg)/180*np.pi    #degrees ˚
    if opt == '--pts':
        xpts = ypts = int(arg)
print("Moire length: ",fs.moire_length(A_1,A_2,theta))

pars_name = "A1="+"{:.3f}".format(A_1).replace('.',',') + "_A2="+"{:.3f}".format(A_2).replace('.',',') + "_theta="+"{:.3f}".format(theta).replace('.',',')+"_pts="+str(xpts)
moire_potential_name = "Data/moire_potential_" + pars_name + ".npy"
#
#
#
grid = 50
#
#
#
try: #Load Moire potential for given Moire lattice
    J = np.load(moire_potential_name)
except:
    command = "python moire.py --A1 "+str(A_1)+" --A2 "+str(A_2)+" --theta "+str(theta)+" --pts "+str(xpts)
    os.system(command)
    J = np.load(moire_potential_name)

#Parameters of moire lattice
A_M = fs.moire_length(A_1,A_2,theta)
a_m1 = A_M*fs.a1
a_m2 = A_M*fs.a2
g_m1 = 2*np.pi/A_M*np.array([1,1/np.sqrt(3)])
g_m2 = 2*np.pi/A_M*np.array([0,2/np.sqrt(3)])
G_M = np.linalg.norm(g_m1)
if 0:   #Plot loaded J
    xpts, ypts = J.shape
    fig, axs = plt.subplots(figsize=(20,20))
    XX = np.linspace(0,1,xpts,endpoint=False)
    dx = XX[1]-XX[0]
    YY = np.linspace(0,1,ypts,endpoint=False)*np.sqrt(3)/2
    a,b = np.meshgrid(XX,YY)
    for i in range(xpts):
        a[i,:] -= 1/2*i*dx
    ax1 = axs.contourf(a,b,J)
    fig.colorbar(ax1)
    plt.show()
    exit()
#Interpolate J
xpts,ypts = J.shape
XX = np.linspace(-2,2,4*xpts,endpoint=False)
YY = np.linspace(-2,2,4*ypts,endpoint=False)
big_J = np.zeros((4*xpts,4*xpts))
big_J[:xpts,:xpts] = J; big_J[:xpts,xpts:2*xpts] = J; big_J[:xpts,2*xpts:3*xpts] = J; big_J[:xpts,3*xpts:] = J;
big_J[xpts:2*xpts,:xpts] = J; big_J[xpts:2*xpts,xpts:2*xpts] = J; big_J[xpts:2*xpts,2*xpts:3*xpts] = J; big_J[xpts:2*xpts,3*xpts:] = J;
big_J[2*xpts:3*xpts,:xpts] = J; big_J[2*xpts:3*xpts,xpts:2*xpts] = J; big_J[2*xpts:3*xpts,2*xpts:3*xpts] = J; big_J[2*xpts:3*xpts,3*xpts:] = J;
big_J[3*xpts:,:xpts] = J; big_J[3*xpts:,xpts:2*xpts] = J; big_J[3*xpts:,2*xpts:3*xpts] = J; big_J[3*xpts:,3*xpts:] = J;
fun_J = RBS(XX,YY,big_J)
if 0:   #plot interpolated moire interlayer function
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
    ax1.set_title("Loaded interlayer coupling")
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
#Extract Fourier components from fun_J to distinguish J' and Phi (in Balents' notation)
from scipy.fft import fft2,fftshift
F = fftshift(fft2(J)/(J.shape[0]*J.shape[1]))
indexes = [ (xpts//2+2,xpts//2),(xpts//2-2,xpts//2),
            (xpts//2,xpts//2+2),(xpts//2,xpts//2-2),
            (xpts//2-2,xpts//2+2),(xpts//2+2,xpts//2-2)]
Jp = 0      #J'
for i in indexes:
    Jp += np.real(F[i[0],i[1]])
Jp /= len(indexes)
Phi0 = F[xpts//2,xpts//2]/Jp   #constant term -> not present in Balents'
print("J' = "+"{:.4f}".format(Jp))
print("Phi0 = "+"{:.4f}".format(Phi0))

def rec_J(x,y):
    #Same function of fun_J in principle
    res = 0
    for i in range(xpts):
        for j in range(xpts):
            res += F[i,j]*np.exp(2*np.pi*1j*((i+xpts)/xpts*x+(j+xpts)/xpts*y))
    return np.real(res)

#Compute the moire potential on the defined grid
X_list = np.linspace(0,1,grid,endpoint=False)
Phi = fun_J(X_list,X_list)/Jp

#Parameters -> Jp and d will form the phase diagram
rho = 1     #Stiffness
d = 0.01     #Anisotropy 

phi_s,phi_a = fs.compute_magnetization(rho,d,Jp,Phi,G_M)

print("Exited minimization")
if 1:   #plot
    fac = 5        #plot 1 arrow every fac sites of grid
    fs.plot_magnetization(phi_s,phi_a,fun_J,fac)
    #Extract magnetization solutions

exit()
if input("Save?(y/N)")=='y':
    np.save(filename_s,phi_s)
    np.save(filename_a,phi_a)














