import numpy as np
import functions as fs
import os,h5py
from pathlib import Path

cluster = fs.get_machine(os.getcwd())


hdf5_name = fs.name_dir_phi(cluster)[:-1]+'.hdf5'
if Path(hdf5_name).is_file():
    os.system('rm '+hdf5_name)
#Open h5py File
with h5py.File(hdf5_name,'w') as f:
    #List elements in directory
    for filename in Path(fs.name_dir_phi(cluster)).iterdir():
        filename = str(filename)
        ds_name = filename[len(filename)-filename[::-1].index('/'):-4]
        if ds_name not in f.keys():
            f.create_dataset(ds_name,data=np.load(filename))





