import numpy as np
import matplotlib.pyplot as plt
import functions as fs

dirname = '/home/dario/Desktop/git/CrBr3/results/'
 
plt.figure(figsize=(15,10))

pts_array = 20
A_M = 100
grid = 500

colors = np.zeros((pts_array,pts_array),dtype=int)
values = fs.compute_grid_pd(pts_array)
cod_col = ['r','pink','g','y','k']
filename_Phi = dirname + 'Phi_'+str(grid)+'_'+str(A_M)+'.npy'
Phi = np.load(filename_Phi)
for i in range(pts_array):
    for j in range(pts_array):
        alpha,beta = values[i,j]
        filename_s = dirname + 'phi_s_' + "{:.4f}".format(alpha) + '_' + "{:.4f}".format(beta) + '.npy'
        filename_a = dirname + 'phi_a_' + "{:.4f}".format(alpha) + '_' + "{:.4f}".format(beta) + '.npy'
        try:
            phi_s = np.load(filename_s)
            phi_a = np.load(filename_a)
            phi_1 = (phi_s+phi_a)/2
            phi_2 = (phi_s-phi_a)/2
            is_1_const = 1 if abs(np.max(phi_1)-np.min(phi_1)) < 1e-1 else 0
            is_2_const = 1 if abs(np.max(phi_2)-np.min(phi_2)) < 1e-1 else 0
            colors[i,j] = is_1_const*2+is_2_const
        except:
            colors[i,j] = 4
        if 0:#colors[i,j] == 0:
            print(alpha,beta)
            fs.plot_magnetization(phi_s,phi_a,Phi)

        plt.scatter(alpha/(1+alpha),beta/(1+beta),color=cod_col[colors[i,j]])
plt.show()

















