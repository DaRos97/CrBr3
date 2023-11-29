import numpy as np
import os,sys
import functions as fs
from pathlib import Path
import inputs

cluster = False
args_general = inputs.args_general
pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = args_general
fs.check_directories(cluster)
#
filename_Phi = fs.name_Phi(cluster)
if not Path(filename_Phi).is_file():
    print("Copying interlayer potential ",filename_Phi)
    os.system('scp rossid@login1.yggdrasil.hpc.unige.ch:'+fs.name_Phi(True)+' '+fs.name_Phi(False))

#Copy .hdf5 file
os.system('scp rossid@login1.yggdrasil.hpc.unige.ch:'+fs.name_dir_phi(True)[:-1]+'.hdf5'+' '+fs.name_dir_phi(False)[:-1]+'.hdf5')
exit()

#Copy all files in the directory
os.system('scp rossid@login1.yggdrasil.hpc.unige.ch:'+fs.name_dir_phi(True)+'* '+fs.name_dir_phi(False))
exit()

values = fs.compute_parameters()
for i in range(pts_array**2):
    alpha,beta = np.reshape(values,(pts_array**2,2))[i]
    filename_phi = fs.name_phi(alpha,beta,False)
    if not Path(filename_phi).is_file():
        os.system('scp rossid@login1.yggdrasil.hpc.unige.ch:'+fs.name_phi(alpha,beta,True)+' '+fs.name_phi(alpha,beta,False))

