import numpy as np
import matplotlib.pyplot as plt
import functions as fs
from scipy.interpolate import RectBivariateSpline as RBS
import sys,getopt

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "M:", ["A1=","A2=","theta="])
    A_1 = 1.0
    A_2 = 1.01
    theta = 0
except:
    print("Error in passed parameters of "+sys.argv[0])
    exit()
for opt, arg in opts:
    if opt == '--A1':
        A_1 = float(arg)
    if opt == '--A2':
        A_2 = float(arg)
    if opt == '--theta':
        theta = float(arg)

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
    #np.save(dataname,data)

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
big_I = np.zeros((2*pts,2*pts))
big_I[:pts,:pts] = I
big_I[pts:,:pts] = I
big_I[:pts,pts:] = I
big_I[pts:,pts:] = I
S_array = np.linspace(-1,1,2*pts,endpoint=False)
fun_I = RBS(S_array,S_array,big_I)

if 0:   #plot interpolated interlayer DFT data
    plt.figure(figsize=(10,10))
    plt.gca().set_aspect('equal')
    nnn = 100
    XX = np.linspace(-1,1,nnn,endpoint=False)
    YY = np.linspace(-1,1,nnn,endpoint=False)*np.sqrt(3)/2
    a,b = np.meshgrid(XX,YY)
    for i in range(nnn):
        a[i,:] -= 1/2*i*(XX[1]-XX[0])
    c,d = np.meshgrid(XX,XX)
    plt.contourf(a,b,fun_I(XX,XX),levels=10)
    plt.xlabel('s1')
    plt.ylabel('s2')
    plt.colorbar()
    plt.show()
    exit()

#Lattice directions -> small a denotes a vector, capital A denotes a distance

#Points to compute of J -> Final Moire potential
xpts = 100
ypts = 100
J = np.zeros((xpts,ypts))
#Lattice points
X = np.linspace(0,1,xpts,endpoint=False)
Y = np.linspace(0,1,ypts,endpoint=False)
#Lattice-1 and lattice-2
xxx = yyy = 100
l1,l2 = fs.compute_lattices(A_1,A_2,theta,xxx,yyy)
#Moire vectors in real and momentum space
A_M = fs.moire_length(A_1,A_2,theta)
a_m1 = A_M*fs.a1
a_m2 = A_M*fs.a2
g_m1 = 2*np.pi/A_M*np.array([1,1/np.sqrt(3)])
g_m2 = 2*np.pi/A_M*np.array([0,2/np.sqrt(3)])

if 1:   #Plot Moirè pattern
    plt.figure(figsize=(10,10))
    plt.gca().set_aspect('equal')
    for n in range(2):
        for y in range(yyy):
            plt.scatter(l1[:,y,n,0],l1[:,y,n,1],color='k',s=10)
            plt.scatter(l2[:,y,n,0],l2[:,y,n,1],color='r',s=10)
    plt.xlim(0,3*A_M)
    plt.ylim(0,3*A_M)
    plt.show()

    #exit()

for i in range(xpts):
    for j in range(ypts):     #Cycle over all considered points in Moirè unit cell
        site = X[i]*a_m1 + Y[j]*a_m2    #x and y components of consider point
        x1,y1,UC = fs.find_closest(l1,A_1,theta,site,'nan')
        x2,y2,UC = fs.find_closest(l2,A_2,theta,site,UC)
        if 0:   #plot two lattices, chosen site and coloured closest sites
            plt.figure(figsize=(10,10))
            plt.gca().set_aspect('equal')
            for n in range(2):
                for y in range(yyy):
                    plt.scatter(l1[:,y,n,0],l1[:,y,n,1],color='k',s=10)
                    plt.scatter(l2[:,y,n,0],l2[:,y,n,1],color='r',s=10)
            plt.scatter(l1[x1,y1,UC,0],l1[x1,y1,UC,1],color='g',s=15)
            plt.scatter(l2[x1,y1,UC,0],l2[x1,y1,UC,1],color='m',s=15)
            plt.scatter(site[0],site[1],color='b',s=20)
            plt.xlim(-A_M/2,A_M)
            plt.ylim(0,A_M*np.sqrt(3)/2)
            plt.show()
            exit()
        #Find displacement
        vec_x1 = np.tensordot(fs.R_z(theta/2),(x1*fs.a1+y1*fs.a2)*A_1 + UC*A_1/np.sqrt(3)*np.array([0,1]),1)
        vec_x2 = np.tensordot(fs.R_z(-theta/2),(x2*fs.a1+y2*fs.a2)*A_2 + UC*A_2/np.sqrt(3)*np.array([0,1]),1)
        d = vec_x1-vec_x2
        S1 = d[0]+d[1]/np.sqrt(3)
        S2 = 2*d[1]/np.sqrt(3)
        #Find value of I[d] and assign it to J[x]
        J[i,j] = fun_I(S1,S2)

if 1:   #Plot found J
    pts = np.zeros((X.shape[0]*Y.shape[0],2))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            pts[i*X.shape[0]+j] = X[i]*a_m1+Y[j]*a_m2
    fig, axs = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(15)
    ax1 = axs.scatter(pts[:,0],pts[:,1],c=np.ravel(J),s=100)
    fig.colorbar(ax1)
    plt.show()

#Save for future use
pars_name = "A1="+"{:.3f}".format(A_1).replace('.',',') + "_A2="+"{:.3f}".format(A_2).replace('.',',') + "_theta="+"{:.3f}".format(theta).replace('.',',')
moire_potential_name = "Data/moire_potential_" + pars_name + ".npy"
np.save(moire_potential_name,J)

























