import numpy as np
import matplotlib.pyplot as plt
import functions as fs

dataname = "Data/CrBr3_interlayer.npy"

#Compute or just load data interlayer
try:
    data = np.load(dataname)
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
    np.save(dataname,data)

#Lattice directions -> small a denotes a vector, capital A denotes a distance
a1 = np.array([1,0])
a2 = np.array([-1/2,np.sqrt(3)/2])

if 0:
    print("integral of I: ",I.sum()/data.shape[0])
    #Plot
    sx = data[:,0]*a1[0] + data[:,1]*a2[0]
    sy = data[:,0]*a1[1] + data[:,1]*a2[1]
    fig, axs = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(15)
    ax1 = axs.scatter(sx,sy,c=data[:,2]-data[:,3],s=100)
    fig.colorbar(ax1)
    axs.set_title("FM-AFM in CrBr3")
    plt.show()

I = data[:,2]-data[:,3]      #FM-AFM

#Points to compute of J
xpts = 80
ypts = 80
J = np.zeros((xpts,ypts))
#Lattice-1 and lattice-2
A_1 = 1
A_2 = 1.1
xxx = yyy = 100
l1 = np.zeros((xxx,yyy,2,2))
l2 = np.zeros((xxx,yyy,2,2))
for i in range(xxx):
    for j in range(yyy):
        cord = i*a1 + j*a2
        l1[i,j,0] = A_1*cord
        l1[i,j,1] = l1[i,j,0] + np.array([0,A_1/np.sqrt(3)])
        l2[i,j,0] = A_2*cord
        l2[i,j,1] = l2[i,j,0] + np.array([0,A_2/np.sqrt(3)])
#Moire vectors in real and momentum space
A_M = A_1*A_2/abs(A_1-A_2)
a_m1 = A_M*a1
a_m2 = A_M*a2
g_m1 = 2*np.pi/A_M*np.array([1,1/np.sqrt(3)])
g_m2 = 2*np.pi/A_M*np.array([0,2/np.sqrt(3)])
#Lattic points
X = np.linspace(0,1,xpts,endpoint=False)
Y = np.linspace(0,1,ypts,endpoint=False)

print("Moire length: ",A_M)
for i in range(xpts):
    for j in range(ypts):     #Cycle over all considered points in Moirè unit cell
        site = X[i]*a_m1 + Y[j]*a_m2
#        print("Site: ",site)
        #Find closest latice site of lattice-1
        x1,y1,UC = fs.find_closest(l1,A_1,site,'nan')
        x2,y2,UC = fs.find_closest(l2,A_2,site,UC)
#        print("Closest lattice-1: ",l1[x1,y1,UC])
#        print("Closest lattice-2: ",l2[x2,y2,UC])
        if 0:#plot two lattices, chosen site and coloured closest sites
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
            plit.show()
        #Find displacement
        vec_x1 = (x1*a1+y1*a2)*A_1 + UC*A_1/np.sqrt(3)*np.array([0,1])
        vec_x2 = (x2*a1+y2*a2)*A_2 + UC*A_2/np.sqrt(3)*np.array([0,1])
#        print("Vectors 1: ",vec_x1)
#        print("Vectors 2: ",vec_x2)
        d = vec_x1-vec_x2
#        print("displacement: ",d)
        S1 = d[0]+d[1]/np.sqrt(3)
        S2 = 2*d[1]/np.sqrt(3)
        #Normalize S1 and S2 to be b/w 0 and 1
#        print("with components ",S1,S2)
        S1 = S1%1 if S1> 0 else 1-abs(S1)%1
        S2 = S2%1 if S2> 0 else 1-abs(S2)%1
        d = S1*a1+S2*a2
#        print("with components ",S1,S2)
        #Find value of I[d] and assign it to J[x]
        J[i,j],index = fs.find_interlayer(data,S1,S2)
#        print(J[i,j],I[index])
        if 0:
            sx = data[:,0]*a1[0] + data[:,1]*a2[0]
            sy = data[:,0]*a1[1] + data[:,1]*a2[1]
            fig, axs = plt.subplots()
            fig.set_figheight(6)
            fig.set_figwidth(15)
            ax1 = axs.scatter(sx,sy,c=data[:,2]-data[:,3],s=100)
            fig.colorbar(ax1)
            axs.scatter(d[0],d[1],color='r',s=100)
            axs.set_title("FM-AFM in CrBr3")
            plt.show()

if 1:   #Plot found J
    #Lattice points in Moirè space
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


























