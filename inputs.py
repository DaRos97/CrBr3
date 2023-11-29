#Input values common to all modules

#Points of the phase diagram to compute
pts_array = 10
pts_gamma = 30
#Points in the moire unit cell
grid = 200
#Points to consider in the moire unit cell when smoothening
pts_per_fit = 2
#Learn rate step
learn_rate_0 = -1e-2
#Moire unit cell
A_M = 20

args_general = (pts_array,pts_gamma,grid,pts_per_fit,learn_rate_0,A_M)

coeff_der = {   
        '1':
            {   '1':(-1,1),
                '2':(-3/2,2,-1/2),
                '3':(-11/6,3,-3/2,1/3),
                '4':(-25/12,4,-3,4/3,-1/4),
            },
        '2':
            {   '1':(1,-2,1),
                '2':(2,-5,4,-1),
                '3':(35/12,-26/3,19/2,-14/3,11/12),
                '4':(15/4,-77/6,107/6,-13,61/12,-5/6),
            },
}
