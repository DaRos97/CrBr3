import numpy as np
import os
import functions as fs
from pathlib import Path

cluster = False
pts_array = 10
grid = 200
pts_per_fit = 31
learn_rate_0 = 1e-2
A_M = 20
args_general = (pts_array,grid,pts_per_fit,learn_rate_0,A_M)
values = fs.compute_grid_pd(pts_array)
#
filename_Phi = fs.name_Phi(grid,A_M,cluster)
if not Path(filename_Phi).is_file():
    print("Copying interlayer potential ",filename_Phi)
    os.system('scp rossid@login1.yggdrasil.hpc.unige.ch:'+fs.name_Phi(grid,A_M,True)+' 'fs.name_Phi(grid,A_M,False))

for i in range(pts_array**2):
    alpha,beta = np.reshape(values,(pts_array**2,2))[i]
    filename_phi = fs.name_phi(alpha,beta,args_general,False)
    if not Path(filename_phi).is_file():
        os.system('scp rossid@login1.yggdrasil.hpc.unige.ch:'fs.name_phi(alpha,beta,args_general,True)+' '+fs.name_phi(alpha,beta,args_general,False))

