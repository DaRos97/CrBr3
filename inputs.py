import numpy as np
#Input values common to all modules

#Points of the phase diagram to compute
pts_array = 20
pts_gamma = 300
#Points in the moire unit cell
grid = 200
#Points to consider in the moire unit cell when smoothening
pts_per_fit = 2
#Learn rate step
learn_rate_0 = -1e-2
#Interlayer potential which depends on Moirè pattern
int_type = 'general'
#int_type = 'basic'
####################################################################
####################################################################
####################################################################
interlayer = {
        'general':{'A_1':1,'A_2':1.01,'theta':0},
        'basic':{'A_M':20}
            }
interlayer['general']['A_M'] = 1/np.sqrt(1/interlayer['general']['A_1']**2+1/interlayer['general']['A_2']**2-2*np.cos(interlayer['general']['theta'])/(interlayer['general']['A_1']*interlayer['general']['A_2']))
A_M = interlayer[int_type]['A_M']
args_general = (pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M)

#Hysteresis parameters
dic_initial_states = {'c+0':0,'c+':42,'t-s':162,'t-a':398}
limit_gamma = 1         #limit value of gamma
steps_gamma = 50        #there will be 2*steps + steps//2 +1 total steps
dic_in_state = ['c+0','c+','t-s','t-a']
