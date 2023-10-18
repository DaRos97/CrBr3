import numpy as np
import functions as fs
import os
from scipy.interpolate import RectBivariateSpline as RBS
import matplotlib.pyplot as plt

A_1 = 1.0
A_2 = 1.01
theta = 0
pars_name = "A1="+"{:.3f}".format(A_1).replace('.',',') + "_A2="+"{:.3f}".format(A_2).replace('.',',') + "_theta="+"{:.3f}".format(theta).replace('.',',')
moire_potential_name = "Data/moire_potential_" + pars_name + ".npy"
grid = 50
try:
    J = np.load(moire_potential_name)
except:
    command = "python moire.py --A1 "+str(A_1)+" --A2 "+str(A_2)+" --theta "+str(theta)
    os.system(command)
    J = np.load(moire_potential_name)

#Parameters moire lattice
A_M = fs.moire_length(A_1,A_2,theta)
a_m1 = A_M*fs.a1
a_m2 = A_M*fs.a2
g_m1 = 2*np.pi/A_M*np.array([1,1/np.sqrt(3)])
g_m2 = 2*np.pi/A_M*np.array([0,2/np.sqrt(3)])
G_M = np.linalg.norm(g_m1)
if 0:   #Plot found J
    X = np.linspace(0,1,J.shape[0],endpoint=False)
    Y = np.linspace(0,1,J.shape[1],endpoint=False)
    pts = np.zeros((J.shape[0]*J.shape[1],2))
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            pts[i*J.shape[0]+j] = X[i]*a_m1+Y[j]*a_m2
    fig, axs = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(15)
    ax1 = axs.scatter(pts[:,0],pts[:,1],c=np.ravel(J),s=100)
    fig.colorbar(ax1)
    plt.show()
    exit()
#Interpolate J
X,Y = J.shape
XX = np.linspace(0,1,X,endpoint=False)
YY = np.linspace(0,1,Y,endpoint=False)
fun_J = RBS(XX,YY,J)
if 0:   #plot interpolated moire interlayer function
    X = Y = 200
    XX = np.linspace(0,1,X,endpoint=False)
    YY = np.linspace(0,1,Y,endpoint=False)*np.sqrt(3)/2
    a,b = np.meshgrid(XX,YY)
    for i in range(X):
        a[i,:] -= 1/2*i*(XX[1]-XX[0])
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.contourf(a,b,fun_J(XX,XX).T,levels=10)
    plt.colorbar()
    plt.show()
    exit()
#Compute the moire potential on the defined grid
Phi = fun_J(np.linspace(0,1,grid),np.linspace(0,1,grid))
#Parameters -> J_p and d will form the phase diagram
rho = 1     #Stiffness
J_p = 0.1   #Interlayer value J'
d = 0.1     #Anisotropy 

alpha = 2*J_p/rho/G_M**2
beta = 2*d/rho/G_M**2

#Start by defining the energy as a function of phi_s and phi_a
#Define initial value of phi_s and phi_a
phi_new_s,phi_new_a = fs.initial_point(A_M)
phi_s = np.zeros((grid,grid))
phi_a = np.zeros((grid,grid))
learn_rate = 0.1
while True:
    if fs.shall_we_exit(phi_s,phi_new_s,phi_a,phi_new_a):
        break
    else:
        phi_s = phi_new_s
        phi_a = phi_new_a
    #
    phi_new_s = phi_s + learn_rate*fs.grad_H(phi_s,phi_a,'s')
    phi_new_a = phi_s + learn_rate*fs.grad_H(phi_s,phi_a,'s')

print("Exited minimization")
#Plot magnetization patterns



















