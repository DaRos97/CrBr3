import numpy as np
import os,sys
import functions as fs
from pathlib import Path
import inputs

dic_clus = {    'y': 'rossid@login1.yggdrasil.hpc.unige.ch:',
                'b': 'rossid@login2.baobab.hpc.unige.ch:',
                'm': 'maf:',
            }
try:
    ind_cluster = sys.argv[1][0]
    data_to_copy = sys.argv[1][1]
except:
    ind_cluster = 'b'       #baobab
    data_to_copy = 'b'      #'b'asic or 'h'ysteresis
cluster_name = 'maf' if ind_cluster == 'm' else 'hpc'

args_general = inputs.args_general
pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M = args_general
#Copy interlayer potential
filename_Phi = fs.name_Phi()
if not Path(filename_Phi).is_file():
    print("Copying interlayer potential ",filename_Phi)
    os.system('scp '+dic_clus[ind_cluster]+fs.name_Phi(cluster_name)+' '+fs.name_Phi())

#Copy result of phase diagram
if data_to_copy=='b':
    #Copy .hdf5 file
    os.system('scp '+dic_clus[ind_cluster]+fs.name_dir_phi(cluster_name)[:-1]+'.hdf5'+' '+fs.name_dir_phi()[:-1]+'.hdf5')
elif data_to_copy=='h':
    for i in range(len(inputs.dic_in_state)):
        os.system('scp '+dic_clus[ind_cluster]+fs.name_hys(inputs.dic_in_state[i],cluster_name)+' '+fs.name_hys(inputs.dic_in_state[i]))






