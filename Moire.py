import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import functions as fs
import inputs,os
from scipy.interpolate import RectBivariateSpline as RBS
from pathlib import Path

xpts = ypts = inputs.grid
cluster = fs.get_machine(os.getcwd())
moire_potential_fn = fs.name_Phi(cluster)

if Path(moire_potential_fn).is_file():
    print("Already computed..")
    fs.plot_Phi(np.load(moire_potential_fn))
    exit()

#print("Moire length: ",fs.moire_length(A_1,A_2,theta))

data_fn = inputs.home_dirname[cluster]+"Data/CrBr3_interlayer.npy"

#Compute or just load data interlayer
if Path(data_fn).is_file():
    I = np.load(data_fn)
else:
    I = fs.compute_dft_data(cluster,save=True)
    np.save(data_fn,I)

#Interpolate interlayer DFT data
pts = I.shape[0]
big_I = fs.extend(I,5)
S_array = np.linspace(-2,3,5*pts,endpoint=False)
fun_I = RBS(S_array,S_array,big_I)

if 1:   #plot interpolated interlayer DFT data
    fs.plot_Phi(I)
    exit()

#Lattice directions -> small a denotes a vector, capital A denotes a distance

#Lattice-1 and lattice-2
l1,l2,xxx,yyy = fs.compute_lattices()
#Moire vectors in real and momentum space
A_M = fs.moire_length(A_1,A_2,theta)
a1 = np.matmul(fs.R_z(theta),fs.a1)
a2 = np.matmul(fs.R_z(theta),fs.a2)
a_m1 = A_M*a1
a_m2 = A_M*a2

if 1:   #Plot Moirè pattern
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
    plt.axis('off')
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
        if i==j and i==-1:   #plot two lattices, chosen site and coloured closest sites
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

























