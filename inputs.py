import numpy as np
#Input values common to all modules

#Points of the phase diagram to compute
pts_array = 20      #alpha and beta
pts_gamma = 1000
#Points in the moire unit cell
grid = 100
#Points to consider in the moire unit cell when smoothening
pts_per_fit = 2
#Learn rate step (needs to be negative)
learn_rate_0 = -1e-2

#Interlayer potential which depends on Moirè pattern
#moire_type = 'general'
#moire_type = 'uniaxial'
moire_type = 'biaxial'
#moire_type = 'shear'

symmetrize = True

####################################################################
####################################################################
####################################################################
moire_pars = {
        'general':{
            'e_xx':0.1,
            'e_yy':0.3,
            'e_xy':0.15,
            },
        'uniaxial':{
            'eps':0.1,
            'ni':0.3,
            'phi':0.15,
            },
        'biaxial':{
            'eps':0.1,
            },
        'shear':{
            'e_xy':0.1,
            'phi':0.3,
            },

        'theta':0.,
        }

args_general = (pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0)


#Home directory names
home_dirname = {
        'loc':'/home/dario/Desktop/git/CrBr3/',
        'hpc':'/home/users/r/rossid/CrBr3/',
        'maf':'/users/rossid/CrBr3/',
        }
