import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import functions as fs
from scipy.interpolate import RectBivariateSpline as RBS
import sys,getopt

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "M:", ["A1=","A2=","theta=","pts="])
    A_1 = 1.0
    A_2 = 1.0
    theta = 0
    xpts = ypys = 100
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
big_I = np.zeros((4*pts,4*pts))
big_I[:pts,:pts] = I; big_I[:pts,pts:2*pts] = I; big_I[:pts,2*pts:3*pts] = I; big_I[:pts,3*pts:] = I;
big_I[pts:2*pts,:pts] = I; big_I[pts:2*pts,pts:2*pts] = I; big_I[pts:2*pts,2*pts:3*pts] = I; big_I[pts:2*pts,3*pts:] = I;
big_I[2*pts:3*pts,:pts] = I; big_I[2*pts:3*pts,pts:2*pts] = I; big_I[2*pts:3*pts,2*pts:3*pts] = I; big_I[2*pts:3*pts,3*pts:] = I;
big_I[3*pts:,:pts] = I; big_I[3*pts:,pts:2*pts] = I; big_I[3*pts:,2*pts:3*pts] = I; big_I[3*pts:,3*pts:] = I;
S_array = np.linspace(-2,2,4*pts,endpoint=False)
fun_I = RBS(S_array,S_array,big_I)

if 0:   #plot interpolated interlayer DFT data
    plt.figure(figsize=(20,20))
    plt.gca().set_aspect('equal')
    #we want each point to be in units of a1, a2
    nnn = 100
    XX = np.linspace(-1,1,nnn,endpoint=False)
    dx = XX[1]-XX[0]
    YY = np.linspace(-1,1,nnn,endpoint=False)*np.sqrt(3)/2
    a,b = np.meshgrid(XX,YY)
    for i in range(nnn):
        a[i,:] -= 1/2*i*dx
    plt.subplot(1,2,1)
    plt.title("Interpolated data")
    plt.contourf(a,b,fun_I(XX,XX),levels=10)
    plt.xlabel('s1')
    plt.ylabel('s2')
    plt.colorbar()
    plt.subplot(1,2,2)
    nnn = 42
    XX = np.linspace(-1,1,nnn,endpoint=False)
    dx = XX[1]-XX[0]
    YY = np.linspace(-1,1,nnn,endpoint=False)*np.sqrt(3)/2
    a,b = np.meshgrid(XX,YY)
    for i in range(nnn):
        a[i,:] -= 1/2*i*dx
    plt.title("Marco's data")
    plt.contourf(a,b,big_I,levels=10)
    plt.xlabel('s1')
    plt.ylabel('s2')
    plt.colorbar()
    plt.show()
    exit()

#Lattice directions -> small a denotes a vector, capital A denotes a distance

#Points to compute of J -> Final Moire potential
J = np.zeros((xpts,ypts))
X = np.linspace(0,1,xpts,endpoint=False)
Y = np.linspace(0,1,ypts,endpoint=False)
#Lattice-1 and lattice-2
l1,l2,xxx,yyy = fs.compute_lattices(A_1,A_2,theta)
#Moire vectors in real and momentum space
A_M = fs.moire_length(A_1,A_2,theta)
a_m1 = A_M*fs.a1
a_m2 = A_M*fs.a2
g_m1 = 2*np.pi/A_M*np.array([1,1/np.sqrt(3)])
g_m2 = 2*np.pi/A_M*np.array([0,2/np.sqrt(3)])

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
#    plt.xlim(-8*A_M,8*A_M)
#    plt.ylim(-2*A_M,6*A_M)
    plt.show()
    exit()

#Compute interlayer energy by evaluating the local stacking of the two layers
for i in tqdm(range(xpts)):
    for j in range(ypts):     #Cycle over all considered points in Moirè unit cell
        site = X[i]*a_m1 + Y[j]*a_m2    #x and y components of consider point
        x1,y1,UC = fs.find_closest(l1,site,'nan')
        x2,y2,UC = fs.find_closest(l2,site,UC)
        if 0:   #plot two lattices, chosen site and coloured closest sites
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
#        vec_x1 = np.tensordot(fs.R_z(theta/2),(x1*fs.a1+y1*fs.a2)*A_1 + UC*A_1/np.sqrt(3)*np.array([0,1]),1)
#        vec_x2 = np.tensordot(fs.R_z(-theta/2),(x2*fs.a1+y2*fs.a2)*A_2 + UC*A_2/np.sqrt(3)*np.array([0,1]),1)
#        d = vec_x1-vec_x2
        S1 = disp[0]+disp[1]/np.sqrt(3)
        S2 = 2*disp[1]/np.sqrt(3)
        #Find value of I[d] and assign it to J[x]
        J[i,j] = fun_I(S1,S2)

ask_print = input("Print found interlayer interaction? (y/N)")
if ask_print == 'y':   #Plot found J
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
#Save for future use
pars_name = "A1="+"{:.3f}".format(A_1).replace('.',',') + "_A2="+"{:.3f}".format(A_2).replace('.',',') + "_theta="+"{:.3f}".format(theta).replace('.',',')+"_pts="+str(xpts)
moire_potential_name = "Data/moire_potential_" + pars_name + ".npy"
np.save(moire_potential_name,J)

print("Saving interlayer interaction for ",pars_name)
























