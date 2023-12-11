import numpy as np
import os,sys
import functions as fs
from pathlib import Path
import inputs

args_general = inputs.args_general
pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = args_general
#Copy interlayer potential
filename_Phi = fs.name_Phi()
if not Path(filename_Phi).is_file():
    print("Copying interlayer potential ",filename_Phi)
    os.system('scp rossid@login2.baobab.hpc.unige.ch:'+fs.name_Phi('hpc')+' '+fs.name_Phi('loc'))

#Copy result of phase diagram
if sys.argv[1]=='b':
    #Copy .hdf5 file
    os.system('scp rossid@login2.baobab.hpc.unige.ch:'+fs.name_dir_phi('hpc')[:-1]+'.hdf5'+' '+fs.name_dir_phi('loc')[:-1]+'.hdf5')
elif sys.argv[1]=='h':
    for i in range(len(inputs.dic_in_state)):
        os.system('scp rossid@login2.baobab.hpc.unige.ch:'+fs.name_hys(inputs.dic_in_state[i],'hpc')+' '+fs.name_hys(inputs.dic_in_state[i],'loc'))
else:
    print("Par not recognized (pd or h)")

exit()






if 0:
    #Copy all files in the directory
    fs.check_directories(cluster)
    os.system('scp rossid@login1.yggdrasil.hpc.unige.ch:'+fs.name_dir_phi(True)+'* '+fs.name_dir_phi(False))
    exit()

if 0:
    fs.check_directories(cluster)
    #Copy by value of parameters
    values = fs.compute_parameters()
    for i in range(10):#pts_array**2):
        gamma,alpha,beta = values[i]
        filename_phi = fs.name_phi(values[i],False)
        if not Path(filename_phi).is_file():
            os.system('scp rossid@login1.yggdrasil.hpc.unige.ch:'+fs.name_phi(values[i],True)+' '+fs.name_phi(values[i],False))

