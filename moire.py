import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import functions as fs
import inputs
from scipy.interpolate import RectBivariateSpline as RBS
from pathlib import Path

A_1 = inputs.interlayer['general']['A_1']
A_2 = inputs.interlayer['general']['A_2']
theta = inputs.interlayer['general']['theta']
xpts = ypts = inputs.grid
moire_potential_name = fs.name_Phi(False)
title = "(A2-A1)/A1 = "+"{:.4f}".format((A_2-A_1)/A_1)+", theta (rad) = "+"{:.4f}".format(theta)
if Path(moire_potential_name).is_file():
    print("Already computed..")
    fs.plot_Phi(np.load(moire_potential_name),title)
    if input("Translate to csv for Ivo? (y/N)")=='y':
        np.savetxt('results/ivo/'+moire_potential_name[41:-4]+'.csv',np.load(moire_potential_name))
    exit()
print("Moire length: ",fs.moire_length(A_1,A_2,theta))

dataname = "Data/CrBr3_interlayer.npy"

#Compute or just load data interlayer
try:
    I = np.load(dataname)
except:
    data_marco = "Data/CrBr3_scan.txt"
    with open(data_marco,'r') as f:
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
        I[ind1,ind2] = (data[i,2]-data[i,3])/2
    #
    np.save(dataname,I)

#Interpolate interlayer DFT data
pts = I.shape[0]
big_I = fs.extend(I,4)
S_array = np.linspace(-2,2,4*pts,endpoint=False)
fun_I = RBS(S_array,S_array,big_I)

if 0:   #plot interpolated interlayer DFT data
    plt.figure(figsize=(20,20))
    plt.gca().set_aspect('equal')
    nnn = 100
    long_X = np.linspace(0,1,nnn,endpoint=False)
    X,Y = np.meshgrid(long_X,long_X)
    X = X-Y/2
    Y = Y/2*np.sqrt(3)
    plt.subplot(1,2,1)
    plt.title("Interpolated data")
    plt.contourf(X,Y,fun_I(long_X,long_X),levels=10)
    plt.xlabel('s1')
    plt.ylabel('s2')
    plt.colorbar()
    #
    plt.subplot(1,2,2)
    nnn = I.shape[0]
    long_X = np.linspace(0,1,nnn,endpoint=False)
    X,Y = np.meshgrid(long_X,long_X)
    X = X-Y/2
    Y = Y/2*np.sqrt(3)
    plt.title("Marco's data")
    plt.contourf(X,Y,I,levels=10)
    plt.xlabel('s1')
    plt.ylabel('s2')
    plt.colorbar()
    plt.show()
    exit()

#Lattice directions -> small a denotes a vector, capital A denotes a distance

#Lattice-1 and lattice-2
l1,l2,xxx,yyy = fs.compute_lattices(A_1,A_2,theta)
#Moire vectors in real and momentum space
A_M = fs.moire_length(A_1,A_2,theta)
a1 = np.matmul(fs.R_z(theta),fs.a1)
a2 = np.matmul(fs.R_z(theta),fs.a2)
a_m1 = A_M*a1
a_m2 = A_M*a2
#g_m1 = 2*np.pi/A_M*np.array([1,1/np.sqrt(3)])
#g_m2 = 2*np.pi/A_M*np.array([0,2/np.sqrt(3)])

if 0:   #Plot Moirè pattern
    plt.figure(figsize=(20,20))
    plt.gca().set_aspect('equal')
    if 0:       #Unit cell lines
        for i in range(8):
            y_y = np.ones(100)*i*A_M*np.sqrt(3)/2
            x_x = np.linspace(-4*A_M,8*A_M,100)
            xp = []
            yp = []
            for j in range(100):
                mul = np.matmul(fs.R_z(theta/2),np.array([x_x[j],y_y[j]]))
                xp.append(mul[0])
                yp.append(mul[1])
            plt.plot(xp,yp,'k',linewidth=0.5)
            #plt.hlines(i*A_M*np.sqrt(3)/2,-4*A_M,8*A_M,'k',linewidth=0.5)
            x_x = np.linspace(i*A_M-4*A_M,i*A_M,100)
            y_y = -x_x*np.sqrt(3)+A_M*i*np.sqrt(3)
            xp = []
            yp = []
            for j in range(100):
                mul = np.matmul(fs.R_z(theta/2),np.array([x_x[j],y_y[j]]))
                xp.append(mul[0])
                yp.append(mul[1])
            plt.plot(xp,yp,'k',linewidth=0.5)
    #
    for n in range(2):      #Actual lattices
        for y in range(yyy):
            plt.scatter(l1[:,y,n,0],l1[:,y,n,1],color='b',s=3)
            plt.scatter(l2[:,y,n,0],l2[:,y,n,1],color='r',s=3)
    plt.title(title)
    plt.show()
    exit()

#Compute interlayer energy by evaluating the local stacking of the two layers
J = np.zeros((xpts,ypts))
X = np.linspace(0,1,xpts,endpoint=False)
Y = np.linspace(0,1,ypts,endpoint=False)
for i in tqdm(range(xpts)):
    for j in range(ypts):     #Cycle over all considered points in Moirè unit cell
        site = X[i]*a_m1 + Y[j]*a_m2    #x and y components of consider point
        x1,y1,UC = fs.find_closest(l1,site,'nan')
        x2,y2,UC = fs.find_closest(l2,site,UC)
        if i==j and 0:   #plot two lattices, chosen site and coloured closest sites
            plt.figure(figsize=(10,10))
            plt.gca().set_aspect('equal')
            for n in range(2):  #lattices
                for y in range(yyy):
                    plt.scatter(l1[:,y,n,0],l1[:,y,n,1],color='b',s=3)
                    plt.scatter(l2[:,y,n,0],l2[:,y,n,1],color='r',s=3)
            plt.scatter(l1[x1,y1,UC,0],l1[x1,y1,UC,1],color='g',s=15)
            plt.scatter(l2[x2,y2,UC,0],l2[x2,y2,UC,1],color='m',s=15)
            plt.scatter(site[0],site[1],color='b',s=20)
#            plt.xlim(-A_M/2,A_M)
#            plt.ylim(0,A_M*np.sqrt(3)/2)
            plt.show()
            exit()
        #Find displacement
        disp = l1[x1,y1,UC] - l2[x2,y2,UC]
        S1 = disp[0]+disp[1]/np.sqrt(3)
        S2 = 2*disp[1]/np.sqrt(3)
        #Find value of I[d] and assign it to J[x]
        J[i,j] = fun_I(S1,S2)
#Smooth
J = fs.smooth(J)[0]

if 1:#input("Print found interlayer interaction? (y/N)")=='y':
    fs.plot_Phi(J)

if input("Save? (y/N)")=='y':
    np.save(moire_potential_name,J)

























