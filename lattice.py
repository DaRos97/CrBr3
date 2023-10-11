import numpy as np
import matplotlib.pyplot as plt

vec1 = np.array([1,0])
vec2 = np.array([-1/2,np.sqrt(3)/2])
a_1 = 1
a_2 = 1.1
xxx = 100
yyy = 100
coord_lim = 50
l1 = np.zeros((xxx,yyy,2,2))
l2 = np.zeros((xxx,yyy,2,2))

for x in range(xxx):
    for y in range(yyy):
        cord = x*vec1 + y*vec2
        if cord[0]*a_1<coord_lim and cord[1]*a_1<coord_lim:
            l1[x,y,0] = a_1*cord
            l1[x,y,1] = l1[x,y,0] + np.array([0,a_1/np.sqrt(3)])
        else:
            l1[x,y,0] = l1[x,y,1] = np.array([np.nan,np.nan])
        if cord[0]*a_2<coord_lim and cord[1]*a_2<coord_lim:
            l2[x,y,0] = a_2*cord
            l2[x,y,1] = l2[x,y,0] + np.array([0,a_2/np.sqrt(3)])
        else:
            l2[x,y,0] = l2[x,y,1] = np.array([np.nan,np.nan])

if 0:
    plt.figure(figsize=(10,10))
    plt.gca().set_aspect('equal')
    for n in range(2):
        for y in range(yyy):
            plt.scatter(l1[:,y,n,0],l1[:,y,n,1],color='k',s=10)
            plt.scatter(l2[:,y,n,0],l2[:,y,n,1],color='r',s=10)
    plt.xlim(0,coord_lim)
    plt.ylim(0,coord_lim)
    plt.show()
