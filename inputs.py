#Input values common to all modules

#Points of the phase diagram to compute
pts_array = 10
#Points in the moire unit cell
grid = 200
#Points to consider in the moire unit cell when smoothening
pts_per_fit = 91
#Learn rate step
learn_rate_0 = -1e-2
#Moire unit cell
A_M = 20


args_general = (pts_array,grid,pts_per_fit,learn_rate_0,A_M)
