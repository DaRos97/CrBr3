import numpy as np
import functions as fs
import os,h5py
from pathlib import Path

cluster = False if os.getcwd()[6:11]=='dario' else True

#Open h5py File
f = h5py.File(fs.name_dir_phi(cluster)[:-1]+'.hdf5','w')    #same name as folder but .hdf5

#List elements in directory
for filename in Path(fs.name_dir_phi(cluster)).iterdir():
    filename = str(filename)
    ds_name = filename[len(filename)-filename[::-1].index('/'):-4]
    f.create_dataset(ds_name,data=np.load(filename))





