import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import functions as fs
import os,h5py,sys
from scipy.interpolate import RectBivariateSpline as RBS
from pathlib import Path

disp = False
#
xpts = ypts = 200
machine = fs.get_machine(os.getcwd())
moire_potential_fn = fs.get_Phi_fn(machine)

if Path(moire_potential_fn).is_file():
    print("Already computed interlayer coupling..")
    if disp:
        with h5py.File(moire_potential_fn,'r') as f:
            J = f['Phi']
            a1_m = f['a1_m']
            a2_m = f['a2_m']
            fs.plot_Phi(J,a1_m,a2_m)
    exit()

I = fs.get_dft_data(machine)

#Interpolate interlayer DFT data
pts = I.shape[0]
big_I = fs.extend(I,5)
S_array = np.linspace(-2,3,5*pts,endpoint=False)
fun_I = RBS(S_array,S_array,big_I)

if 0:   #plot interpolated interlayer DFT data
    fs.plot_Phi(I,fs.a1,fs.a2)
    exit()

#Lattice directions -> small a denotes a vector, capital A denotes a distance

#Lattice-1 and lattice-2
l1,l2,a1,a2 = fs.compute_lattices()
a1_m = a1 if a1[0]>a2[0] else a2
a2_m = a2 if a1[0]>a2[0] else a1
print("|a_1|=",np.linalg.norm(a1_m),", |a_2|=",np.linalg.norm(a2_m))

if disp:   #Plot Moirè pattern
    fig,ax = plt.subplots(figsize=(20,20))
    ax.set_aspect('equal')
    #
    for n in range(2):      #sublattice index
        for y in range(l1.shape[1]):
            ax.scatter(l1[:,y,n,0],l1[:,y,n,1],color='b',s=3)
            ax.scatter(l2[:,y,n,0],l2[:,y,n,1],color='r',s=3)

    ax.arrow(0,0,a1_m[0],a1_m[1],color='g',lw=2,head_width=0.2,zorder=-1,alpha=0.5)
    ax.arrow(0,0,a2_m[0],a2_m[1],color='g',lw=2,head_width=0.2,zorder=-1,alpha=0.5)
    ax.axis('off')
    plt.show()
    #exit()

#Compute interlayer energy by evaluating the local stacking of the two layers
J = np.zeros((xpts,ypts))
X = np.linspace(0,1,xpts,endpoint=False)
Y = np.linspace(0,1,ypts,endpoint=False)
for i in tqdm(range(xpts)):
    for j in range(ypts):     #Cycle over all considered points in Moirè unit cell
        site = X[i]*a1_m + Y[j]*a2_m    #x and y components of consider point
        x1,y1,UC = fs.find_closest(l1,site,'nan')
        x2,y2,UC = fs.find_closest(l2,site,UC)
        if i==j and i==-1:   #plot two lattices, chosen site and coloured closest sites
            plt.figure(figsize=(10,10))
            plt.gca().set_aspect('equal')
            for n in range(2):  #lattices
                for y in range(l1.shape[1]):
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
        displacement = l1[x1,y1,UC] - l2[x2,y2,UC]
        S1 = displacement[0]+displacement[1]/np.sqrt(3)
        S2 = 2*displacement[1]/np.sqrt(3)
        #Find value of I[d] and assign it to J[x]
        J[i,j] = fun_I(S1,S2)
#Smooth
J = fs.smooth(J,2,(a1_m,a2_m))[0]

if disp:#input("Print found interlayer interaction? (y/N)")=='y':
   fs.plot_Phi(J,a1_m,a2_m)

if (disp and input("Save? (y/N)")=='y') or not disp:
    with h5py.File(moire_potential_fn,'w') as f:
        f.create_dataset('Phi',data=J)
        f.create_dataset('a1_m',data=a1_m)
        f.create_dataset('a2_m',data=a2_m)


























