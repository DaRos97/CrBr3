import numpy as np
#Input values common to all modules

#Points of the phase diagram to compute
pts_array = 20      #alpha and beta
pts_gamma = 1000
#Points in the moire unit cell
grid = 200
#Points to consider in the moire unit cell when smoothening
pts_per_fit = 2
#Learn rate step (needs to be negative)
learn_rate_0 = -1e-2
#Interlayer potential which depends on Moirè pattern
int_type = 'general'
#int_type = 'basic'

symmetrize = True

####################################################################
####################################################################
####################################################################
interlayer = {
        'general':{
            'e_xx':1,
            'e_yy':1,
            'e_xy':1,
            'theta':0
            },
        'basic':{'A_M':20}
            }
args_general = (pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0)


#Home directory names
home_dirname = {
        'loc':'/home/dario/Desktop/git/CrBr3/',
        'hpc':'/home/users/r/rossid/CrBr3/',
        'maf':'/users/rossid/CrBr3/',
        }
