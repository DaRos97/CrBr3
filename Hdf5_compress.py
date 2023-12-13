import numpy as np
import functions as fs
import os,h5py
from pathlib import Path
from time import time

t0 = time()
cluster = fs.get_machine(os.getcwd())


hdf5_name = fs.name_dir_phi(cluster)[:-1]+'.hdf5'
print(hdf5_name)
if not cluster == 'loc':
    if Path(hdf5_name).is_file():
        os.system('rm '+hdf5_name)
#Open h5py File
with h5py.File(hdf5_name,'a') as f:
    #List elements in directory
    for filename in Path(fs.name_dir_phi(cluster)).iterdir():
        filename = str(filename)
        ds_name = filename[len(filename)-filename[::-1].index('/'):-4]
        #if ds_name not in f.keys():
        try:
            f.create_dataset(ds_name,data=np.load(filename))
        except:
            print(filename)

print('Time taken: ',time()-t0)



