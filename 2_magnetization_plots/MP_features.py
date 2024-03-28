import numpy as np
import functions as fs

nnnn = np.nan
"""
Here we report the field at which features happen. M_bi refers only to flop transitions because are the only ones happening.
AA is divided in flip and flop. 'nan' means is not the transition happening. A field of 0 in flip means the starting state
is already aligned, while a field of 0 in flop means the starting state is flopped.
"""

"""
Biaxial strain
"""
M_bi = np.zeros((6,5,5))
AA_bi = np.zeros((6,5,5))
AA_bi_flip = np.zeros((6,5,5))
AA_bi_flop = np.zeros((6,5,5))
"""
5% strain
"""
M_bi[0] = [ [0.52,   0.90,   1.36,   1.66,   1.92],
            [0.46,   0.84,   1.28,   1.56,   1.80],
            [0.36,   0.74,   1.18,   1.46,   1.68],
            [0.20,   0.66,   1.10,   1.36,   1.58],
            [0.00,   0.00,   0.26,   0.54,   0.74],
            ]
AA_bi[0] = [        [0.10,   0.14,   0.12,   0.10,   0.10],
                    [0.00,   0.02,   0.00,   0.00,   0.00],
                    [0.00,   0.00,   0.00,   0.00,   0.00],
                    [0.00,   0.00,   0.00,   0.00,   0.00],
                    [0.00,   0.00,   0.00,   0.00,   0.00],
                    ]
AA_bi_flip[0] = [   [nnnn,   nnnn,   0.12,   0.10,   0.10],
                    [nnnn,   nnnn,   0.00,   0.00,   0.00],
                    [nnnn,   nnnn,   0.00,   0.00,   0.00],
                    [nnnn,   0.00,   0.00,   0.00,   0.00],
                    [0.00,   0.00,   0.00,   0.00,   0.00],
                    ]
AA_bi_flop[0] = [   [0.10,   0.14,   nnnn,   nnnn,   nnnn],
                    [0.00,   0.02,   nnnn,   nnnn,   nnnn],
                    [0.00,   0.00,   nnnn,   nnnn,   nnnn],
                    [0.00,   nnnn,   nnnn,   nnnn,   nnnn],
                    [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],
                    ]
"""
4% strain
"""
M_bi[1] = [ [0.54,   0.92,   1.38,   1.68,   1.92],
            [0.50,   0.86,   1.32,   1.62,   1.84],
            [0.44,   0.82,   1.26,   1.54,   1.76],
            [0.38,   0.76,   1.20,   1.48,   1.70],
            [0.06,   0.22,   0.74,   1.00,   1.20],
            ]
AA_bi[1] = [        [0.12,   0.16,   0.14,   0.14,   0.12],
                    [0.06,   0.08,   0.04,   0.00,   0.00],
                    [0.00,   0.00,   0.00,   0.00,   0.00],
                    [0.00,   0.00,   0.00,   0.00,   0.00],
                    [0.00,   0.00,   0.00,   0.00,   0.00],
                    ]
AA_bi_flip[1] = [   [nnnn,   nnnn,   0.14,   0.14,   0.12],
                    [nnnn,   nnnn,   0.04,   0.00,   0.00],
                    [nnnn,   nnnn,   0.00,   0.00,   0.00],
                    [nnnn,   nnnn,   0.00,   0.00,   0.00],
                    [0.00,   0.00,   0.00,   0.00,   0.00],
                    ]

AA_bi_flop[1] = [   [0.12,   0.16,   nnnn,   nnnn,   nnnn],
                    [0.06,   0.08,   nnnn,   nnnn,   nnnn],
                    [0.00,   0.00,   nnnn,   nnnn,   nnnn],
                    [0.00,   0.00,   nnnn,   nnnn,   nnnn],
                    [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],
                    ]
"""
3% strain
"""
M_bi[2] = [ [0.54,   0.92,   1.40,   1.70,   1.94],
            [0.52,   0.90,   1.36,   1.66,   1.90],
            [0.48,   0.86,   1.32,   1.60,   1.84],
            [0.46,   0.84,   1.28,   1.58,   1.80],
            [0.16,   0.60,   1.06,   1.32,   1.52],
            ]
AA_bi[2] = [        [0.12,   0.16,   0.16,   0.14,   0.12],
                    [0.10,   0.12,   0.10,   0.08,   0.06],
                    [0.06,   0.06,   0.02,   0.00,   0.00],
                    [0.02,   0.00,   0.00,   0.00,   0.00],
                    [0.00,   0.00,   0.00,   0.00,   0.00],
                    ]
AA_bi_flip[2] = [   [nnnn,   nnnn,   0.16,   0.14,   0.12],
                    [nnnn,   nnnn,   0.10,   0.08,   0.06],
                    [nnnn,   nnnn,   0.02,   0.00,   0.00],
                    [nnnn,   nnnn,   0.00,   0.00,   0.00],
                    [0.00,   0.00,   0.00,   0.00,   0.00],
                    ]
AA_bi_flop[2] = [   [0.12,   0.16,   nnnn,   nnnn,   nnnn],
                    [0.10,   0.12,   nnnn,   nnnn,   nnnn],
                    [0.06,   0.06,   nnnn,   nnnn,   nnnn],
                    [0.02,   0.00,   nnnn,   nnnn,   nnnn],
                    [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],
                    ]
"""
2% strain
"""
M_bi[3] = [ [0.54,   0.92,   1.40,   1.72,   1.94],
            [0.54,   0.92,   1.38,   1.68,   1.94],
            [0.52,   0.90,   1.36,   1.66,   1.90],
            [0.52,   0.88,   1.34,   1.64,   1.88],
            [0.42,   0.80,   1.24,   1.52,   1.74],
            ]
AA_bi[3] = [        [0.14,   0.20,   0.16,   0.14,   0.12],
                    [0.12,   0.16,   0.16,   0.14,   0.12],
                    [0.10,   0.14,   0.10,   0.08,   0.08],
                    [0.08,   0.10,   0.08,   0.06,   0.04],
                    [0.00,   0.00,   0.00,   0.00,   0.00],
            ]
AA_bi_flip[3] = [   [nnnn,   nnnn,   0.16,   0.14,   0.12],
                    [nnnn,   nnnn,   0.16,   0.14,   0.12],
                    [nnnn,   nnnn,   0.10,   0.08,   0.08],
                    [nnnn,   nnnn,   0.08,   0.06,   0.04],
                    [nnnn,   nnnn,   0.00,   0.00,   0.00],
                    ]
AA_bi_flop[3] = [   [0.14,   0.20,   nnnn,   nnnn,   nnnn],
                    [0.12,   0.16,   nnnn,   nnnn,   nnnn],
                    [0.10,   0.14,   nnnn,   nnnn,   nnnn],
                    [0.08,   0.10,   nnnn,   nnnn,   nnnn],
                    [0.00,   0.00,   nnnn,   nnnn,   nnnn],
                    ]
"""
1% strain
"""
M_bi[4] = [ [0.54,   0.94,   1.40,   1.72,   1.94],
            [0.54,   0.94,   1.40,   1.72,   1.94],
            [0.54,   0.92,   1.40,   1.70,   1.94],
            [0.54,   0.92,   1.40,   1.70,   1.94],
            [0.52,   0.90,   1.36,   1.66,   1.90],
            ]
AA_bi[4] = [        [0.14,   0.22,   0.16,   0.14,   0.12],
                    [0.14,   0.22,   0.16,   0.14,   0.12],
                    [0.12,   0.18,   0.16,   0.14,   0.12],
                    [0.12,   0.16,   0.16,   0.14,   0.12],
                    [0.10,   0.12,   0.10,   0.08,   0.06],
                    ]
AA_bi_flip[4] = [   [nnnn,   nnnn,   0.16,   0.14,   0.12],
                    [nnnn,   nnnn,   0.16,   0.14,   0.12],
                    [nnnn,   nnnn,   0.16,   0.14,   0.12],
                    [nnnn,   nnnn,   0.16,   0.14,   0.12],
                    [nnnn,   nnnn,   0.10,   0.08,   0.06],
                    ]
AA_bi_flop[4] = [   [0.14,   0.22,   nnnn,   nnnn,   nnnn],
                    [0.14,   0.22,   nnnn,   nnnn,   nnnn],
                    [0.12,   0.18,   nnnn,   nnnn,   nnnn],
                    [0.12,   0.16,   nnnn,   nnnn,   nnnn],
                    [0.10,   0.12,   nnnn,   nnnn,   nnnn],
                    ]
"""
0.5% strain
"""
M_bi[5] = [ [0.54,   0.94,   1.42,   1.72,   1.96],
            [0.54,   0.94,   1.42,   1.72,   1.96],
            [0.54,   0.94,   1.40,   1.72,   1.94],
            [0.54,   0.94,   1.40,   1.72,   1.94],
            [0.52,   0.92,   1.40,   1.70,   1.92],
            ]
AA_bi[5] = [        [0.14,   0.22,   0.16,   0.14,   0.12],
                    [0.14,   0.22,   0.16,   0.14,   0.12],
                    [0.14,   0.22,   0.16,   0.14,   0.12],
                    [0.14,   0.22,   0.16,   0.14,   0.12],
                    [0.12,   0.18,   0.16,   0.14,   0.12],
                    ]
AA_bi_flip[5] = [   [nnnn,   nnnn,   0.16,   0.14,   0.12],
                    [nnnn,   nnnn,   0.16,   0.14,   0.12],
                    [nnnn,   nnnn,   0.16,   0.14,   0.12],
                    [nnnn,   nnnn,   0.16,   0.14,   0.12],
                    [nnnn,   nnnn,   0.16,   0.14,   0.12],
                    ]
AA_bi_flop[5] = [   [0.14,   0.22,   nnnn,   nnnn,   nnnn],
                    [0.14,   0.22,   nnnn,   nnnn,   nnnn],
                    [0.14,   0.22,   nnnn,   nnnn,   nnnn],
                    [0.14,   0.22,   nnnn,   nnnn,   nnnn],
                    [0.12,   0.18,   nnnn,   nnnn,   nnnn],
                    ]


"""
Uniaxial strain
"""
M_uni = np.zeros((6,6,5,5))
AA_uni = np.zeros((6,6,5,5))
AA_uni_flip = np.zeros((6,6,5,5))
AA_uni_flop = np.zeros((6,6,5,5))
"""
5% strain - tr 0
"""
M_uni[0,0] = [ 
            [0.54,   0.92,   1.38,   1.70,   1.94],
            [0.50,   0.88,   1.32,   1.62,   1.86],
            [0.42,   0.82,   1.26,   1.54,   1.78],
            [0.36,   0.76,   1.20,   1.48,   1.70],
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
AA_uni_flip[0,0] = [   
            [nnnn,   nnnn,   0.14,   0.14,   0.14],
            [nnnn,   nnnn,   0.06,   0.04,   0.02],
            [nnnn,   nnnn,   0.00,   0.00,   0.00],
            [nnnn,   nnnn,   0.00,   0.00,   0.00],
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
AA_uni_flop[0,0] = [   
            [0.12,   0.16,   nnnn,   nnnn,   nnnn],
            [0.04,   0.08,   nnnn,   nnnn,   nnnn],
            [0.00,   0.00,   nnnn,   nnnn,   nnnn],
            [0.00,   0.00,   nnnn,   nnnn,   nnnn],
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
AA_uni[0,0] = [        
            [0.12,   0.16,   0.14,   0.14,   0.14],
            [0.04,   0.08,   0.06,   0.04,   0.02],
            [0.00,   0.00,   0.00,   0.00,   0.00],
            [0.00,   0.00,   0.00,   0.00,   0.00],
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
"""
5% strain - tr 0.1
"""
M_uni[0,1] = [ 
            [0.44,   0.76,   1.12,   1.36,   1.54],
            [0.40,   0.70,   1.06,   1.30,   1.46],
            [0.34,   0.66,   1.00,   1.22,   1.38],
            [0.26,   0.60,   0.94,   1.14,   1.30],
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
AA_uni_flip[0,1] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[0,1] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni[0,1] = [ [nnnn for i in range(5)] for i in range(5)]
"""
5% strain - tr 0.2
"""
M_uni[0,2] = [ 
            [0.12,   0.18,   0.16,   0.16,   0.16],
            [0.08,   0.12,   0.10,   0.08,   0.06],
            [0.02,   0.06,   0.04,   0.00,   0.00],
            [0.00,   0.00,   0.00,   0.00,   0.00],
            [0.00,   0.00,   0.00,   0.00,   0.00],
            ]
AA_uni_flip[0,2] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[0,2] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni[0,2] = [ [nnnn for i in range(5)] for i in range(5)]
"""
5% strain - tr 0.333
"""
M_uni[0,3] = [ 
            [0.30,   0.52,   0.74,   0.86,   0.92],
            [0.26,   0.46,   0.68,   0.78,   0.84],
            [0.20,   0.42,   0.60,   0.70,   0.74],
            [0.14,   0.36,   0.54,   0.62,   0.68],
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
AA_uni_flip[0,3] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[0,3] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni[0,3] = [ [nnnn for i in range(5)] for i in range(5)]
"""
5% strain - tr 0.4      #Here the AA_flop is actually the flop of a portion of the M region
"""
M_uni[0,4] = [ 
            [0.48,   0.82,   1.22,   1.48,   1.70],
            [0.44,   0.78,   1.16,   1.42,   1.62],
            [0.38,   0.72,   1.10,   1.34,   1.54],                 
            [0.32,   0.68,   nnnn,   1.28,   nnnn],                 ####
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
AA_uni_flip[0,4] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[0,4] = [   
            [0.36,   0.62,   0.90,   1.08,   1.20],
            [0.32,   0.56,   0.84,   1.00,   1.12],
            [0.26,   0.52,   0.78,   0.92,   1.02],
            [0.24,   0.46,   0.70,   0.84,   0.94], 
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
AA_uni[0,4] = np.copy(AA_uni_flop[0,4])
"""
5% strain - tr 0.5
"""
M_uni[0,5] = [ 
            [0.52,   0.90,   1.36,   1.68,   1.92],
            [0.48,   0.86,   1.30,   1.60,   1.84],
            [0.44,   0.82,   1.24,   1.54,   1.76],                 
            [0.38,   0.76,   1.20,   1.48,   nnnn],                 ####
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
AA_uni_flip[0,5] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[0,5] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni[0,5] = [ [nnnn for i in range(5)] for i in range(5)]
"""
4% strain - tr 0
"""
M_uni[1,0] = [ 
            [0.54,   0.92,   1.40,   1.70,   1.96],
            [0.52,   0.90,   1.36,   1.66,   1.90],
            [0.48,   0.86,   1.30,   1.60,   1.84],
            [0.44,   0.82,   1.28,   1.56,   1.80],
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
AA_uni_flip[1,0] = [   
            [nnnn,   nnnn,   0.16,   0.16,   nnnn],
            [nnnn,   nnnn,   0.10,   0.08,   0.08],
            [nnnn,   nnnn,   0.04,   0.02,   0.00],
            [nnnn,   nnnn,   0.00,   0.00,   0.00],
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
AA_uni_flop[1,0] = [   
            [0.12,   0.18,   nnnn,   nnnn,   nnnn],
            [0.08,   0.12,   nnnn,   nnnn,   nnnn],
            [0.04,   0.06,   nnnn,   nnnn,   nnnn],
            [0.00,   0.02,   nnnn,   nnnn,   nnnn],
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
AA_uni[1,0] = [        
            [0.12,   0.18,   0.16,   0.16,   nnnn],
            [0.08,   0.12,   0.10,   0.08,   0.08],
            [0.04,   0.06,   0.04,   0.02,   0.00],
            [0.00,   0.02,   0.00,   0.00,   0.00],
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
"""
4% strain - tr 0.1
"""
M_uni[1,1] = [ 
            [0.44,   0.76,   1.14,   1.38,   1.56],
            [0.42,   0.72,   1.10,   1.32,   1.50],
            [0.38,   0.70,   1.06,   1.28,   1.44],
            [0.36,   0.63,   1.02,   1.24,   1.40],
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
AA_uni_flip[1,1] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[1,1] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni[1,1] = [ [nnnn for i in range(5)] for i in range(5)]
"""
4% strain - tr 0.2
"""
M_uni[1,2] = [ 
            [0.12,   0.18,   0.18,   0.16,   0.06],
            [0.10,   0.14,   0.12,   0.12,   0.10],
            [0.06,   0.10,   0.08,   0.06,   0.04],
            [0.04,   0.06,   0.04,   0.02,   0.00],
            [0.00,   0.00,   0.00,   0.00,   0.00],
            ]
AA_uni_flip[1,2] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[1,2] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni[1,2] = [ [nnnn for i in range(5)] for i in range(5)]
"""
4% strain - tr 0.333
"""
M_uni[1,3] = [ 
            [0.32,   0.52,   0.74,   0.86,   0.94],
            [0.28,   0.48,   0.70,   0.82,   0.88],
            [0.26,   0.46,   0.66,   0.76,   0.82],
            [0.22,   0.42,   0.62,   0.72,   0.66],
            [nnnn,   0.16,   0.34,   0.40,   0.44],                 ####
            ]
AA_uni_flip[1,3] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[1,3] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni[1,3] = [ [nnnn for i in range(5)] for i in range(5)]
"""
4% strain - tr 0.4      #Here the AA_flop is actually the flop of a portion of the M region
"""
M_uni[1,4] = [ 
            [0.48,   0.82,   1.24,   1.50,   1.70],
            [0.46,   0.80,   1.20,   1.46,   1.66],
            [0.42,   0.76,   1.16,   1.40,   1.60], 
            [0.40,   0.74,   1.12,   1.36,   1.56], 
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
AA_uni_flip[1,4] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[1,4] = [   
            [0.36,   0.62,   0.92,   1.08,   1.20],
            [0.34,   0.60,   0.88,   1.04,   1.16],
            [0.32,   0.56,   0.82,   0.98,   1.10],
            [0.28,   0.52,   0.78,   0.94,   1.04],                 
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
AA_uni[1,4] = np.copy(AA_uni_flop[1,4])
"""
4% strain - tr 0.5
"""
M_uni[1,5] = [ 
            [0.54,   0.92,   1.38,   1.68,   1.92],
            [0.50,   0.88,   1.34,   1.64,   1.88],
            [0.48,   0.84,   1.30,   1.58,   1.82], 
            [0.44,   0.82,   1.26,   1.54,   1.78], 
            [nnnn,   nnnn,   nnnn,   nnnn,   nnnn],                 ####
            ]
AA_uni_flip[1,5] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[1,5] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni[1,5] = [ [nnnn for i in range(5)] for i in range(5)]
"""
3% strain - tr 0
"""
M_uni[2,0] = [ 
            [0.54,   0.94,   1.40,   1.72,   1.96],
            [0.54,   0.92,   1.38,   1.68,   1.92],
            [0.52,   0.90,   1.36,   1.66,   1.90],
            [0.50,   0.88,   1.32,   1.62,   1.86],
            [0.32,   0.74,   1.18,   1.46,   1.64], 
            ]
AA_uni_flip[2,0] = [   
            [nnnn,   nnnn,   0.16,   0.14,   0.12],
            [nnnn,   nnnn,   0.14,   0.12,   0.12],
            [nnnn,   nnnn,   0.10,   0.08,   0.06],
            [nnnn,   nnnn,   0.06,   0.04,   0.02],
            [nnnn,   nnnn,   0.00,   0.00,   0.00], 
            ]
AA_uni_flop[2,0] = [   
            [0.12,   0.18,   nnnn,   nnnn,   nnnn],
            [0.10,   0.14,   nnnn,   nnnn,   nnnn],
            [0.08,   0.12,   nnnn,   nnnn,   nnnn],
            [0.06,   0.08,   nnnn,   nnnn,   nnnn],
            [0.00,   0.00,   nnnn,   nnnn,   nnnn],                 
            ]
AA_uni[2,0] = [        
            [0.12,   0.18,   0.16,   0.14,   0.12],
            [0.10,   0.14,   0.14,   0.12,   0.12],
            [0.08,   0.12,   0.10,   0.08,   0.06],
            [0.06,   0.08,   0.06,   0.04,   0.02],
            [0.00,   0.00,   0.00,   0.00,   0.00], 
            ]
"""
3% strain - tr 0.1
"""
M_uni[2,1] = [ 
            [0.46,   0.76,   1.14,   1.38,   1.58],
            [0.44,   0.74,   1.12,   1.36,   1.54],
            [0.42,   0.72,   1.10,   1.32,   1.50],
            [0.40,   0.70,   1.06,   1.30,   1.48],
            [0.22,   0.56,   0.90,   1.10,   1.26],
            ]
AA_uni_flip[2,1] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[2,1] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni[2,1] = [ [nnnn for i in range(5)] for i in range(5)]
"""
3% strain - tr 0.2
"""
M_uni[2,2] = [ 
            [0.14,   0.20,   0.16,   0.16,   0.14],
            [0.12,   0.16,   0.16,   0.14,   0.14],
            [0.10,   0.14,   0.12,   0.12,   0.10],
            [0.08,   0.12,   0.10,   0.08,   0.06],
            [0.18,   0.08,   0.00,   0.00,   0.00],
            ]
AA_uni_flip[2,2] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[2,2] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni[2,2] = [ [nnnn for i in range(5)] for i in range(5)]
"""
3% strain - tr 0.333
"""
M_uni[2,3] = [ 
            [0.32,   0.52,   0.76,   0.88,   0.96],
            [0.30,   0.50,   0.72,   0.84,   0.92],
            [0.28,   0.48,   0.70,   0.82,   0.88],
            [0.28,   0.48,   0.68,   0.80,   0.84],
            [0.12,   0.34,   0.52,   0.60,   0.64], 
            ]
AA_uni_flip[2,3] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[2,3] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni[2,3] = [ [nnnn for i in range(5)] for i in range(5)]
"""
3% strain - tr 0.4      #Here the AA_flop is actually the flop of a portion of the M region
"""
M_uni[2,4] = [ 
            [0.48,   0.82,   1.24,   1.50,   1.70],
            [0.48,   0.80,   1.22,   1.48,   1.68],
            [0.46,   0.78,   1.20,   1.46,   1.66], 
            [0.44,   0.78,   1.16,   1.42,   1.62],
            [0.28,   nnnn,   1.02,   1.26,   1.44],                 ####
            ]
AA_uni_flip[2,4] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[2,4] = [   
            [0.38,   0.62,   0.92,   1.10,   1.22],
            [0.36,   0.60,   0.90,   1.06,   1.18],
            [0.34,   0.58,   0.86,   1.04,   1.14],
            [0.32,   0.58,   0.84,   1.00,   1.12],                 
            [0.24,   0.44,   0.68,   0.82,   0.90], 
            ]
AA_uni[2,4] = np.copy(AA_uni_flop[2,4])
"""
3% strain - tr 0.5
"""
M_uni[2,5] = [ 
            [0.54,   0.92,   1.38,   1.70,   1.94],
            [0.52,   0.90,   1.36,   1.66,   1.90],     
            [0.50,   0.88,   1.34,   1.64,   1.86], 
            [0.50,   0.86,   1.32,   1.60,   1.84], 
            [0.34,   0.74,   1.16,   1.44,   1.66], 
            ]
AA_uni_flip[2,5] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[2,5] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni[2,5] = [ [nnnn for i in range(5)] for i in range(5)]
"""
2% strain - tr 0
"""
M_uni[3,0] = [ 
            [0.54,   0.94,   1.40,   1.72,   1.94],
            [0.54,   0.92,   1.40,   1.70,   1.96],
            [0.54,   0.92,   1.38,   1.68,   1.94],
            [0.52,   0.90,   1.38,   1.68,   1.92],
            [0.46,   0.86,   1.30,   1.60,   1.82], 
            ]
AA_uni_flip[3,0] = [   
            [nnnn,   nnnn,   0.16,   0.14,   0.12],
            [nnnn,   nnnn,   0.16,   0.16,   0.12],
            [nnnn,   nnnn,   0.14,   0.14,   0.12],
            [nnnn,   nnnn,   0.12,   0.12,   0.10],
            [nnnn,   nnnn,   0.02,   0.00,   0.00], 
            ]
AA_uni_flop[3,0] = [   
            [0.14,   0.18,   nnnn,   nnnn,   nnnn],
            [0.12,   0.18,   nnnn,   nnnn,   nnnn],
            [0.12,   0.16,   nnnn,   nnnn,   nnnn],
            [0.10,   0.14,   nnnn,   nnnn,   nnnn],
            [0.02,   0.04,   nnnn,   nnnn,   nnnn],                 
            ]
AA_uni[3,0] = [        
            [0.14,   0.18,   0.16,   0.14,   0.12],
            [0.12,   0.18,   0.16,   0.16,   0.12],
            [0.12,   0.16,   0.14,   0.14,   0.12],
            [0.10,   0.14,   0.12,   0.12,   0.10],
            [0.02,   0.04,   0.02,   0.00,   0.00], 
            ]
"""
2% strain - tr 0.1
"""
M_uni[3,1] = [ 
            [0.46,   0.78,   1.16,   1.40,   1.58],
            [0.44,   0.76,   1.14,   1.38,   1.56],
            [0.44,   0.76,   1.12,   1.36,   1.54],
            [0.44,   0.74,   1.12,   1.34,   1.52],
            [0.38,   0.68,   1.04,   1.26,   nnnn],                 ####
            ]
AA_uni_flip[3,1] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[3,1] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni[3,1] = [ [nnnn for i in range(5)] for i in range(5)]
"""
2% strain - tr 0.2
"""
M_uni[3,2] = [ 
            [0.14,   0.20,   0.18,   0.16,   0.14],
            [0.14,   0.18,   0.18,   0.16,   0.14],
            [0.12,   0.18,   0.16,   0.16,   0.14],
            [0.12,   0.16,   0.14,   0.14,   0.12],
            [0.06,   0.08,   0.06,   0.04,   0.04],
            ]
AA_uni_flip[3,2] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[3,2] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni[3,2] = [ [nnnn for i in range(5)] for i in range(5)]
"""
2% strain - tr 0.333
"""
M_uni[3,3] = [ 
            [0.32,   0.52,   0.76,   0.90,   0.96],
            [0.32,   0.52,   0.74,   0.88,   0.94],
            [0.30,   0.50,   0.74,   0.86,   0.92],
            [0.30,   0.50,   0.72,   0.84,   0.90],
            [0.26,   0.44,   0.66,   0.76,   0.80], 
            ]
AA_uni_flip[3,3] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[3,3] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni[3,3] = [ [nnnn for i in range(5)] for i in range(5)]
"""
2% strain - tr 0.4      #Here the AA_flop is actually the flop of a portion of the M region
"""
M_uni[3,4] = [ 
            [0.48,   0.84,   1.24,   1.52,   1.70],
            [0.48,   0.82,   1.24,   1.50,   1.70], 
            [0.48,   0.82,   1.22,   1.48,   1.70], 
            [0.46,   0.80,   1.22,   1.48,   1.68],
            [0.42,   0.76,   nnnn,   nnnn,   1.58],                 ####
            ]
AA_uni_flip[3,4] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[3,4] = [   
            [0.38,   0.62,   0.92,   1.10,   1.28],
            [0.36,   0.62,   0.92,   1.08,   1.22],
            [0.36,   0.62,   0.90,   1.08,   1.20],
            [0.36,   0.60,   0.88,   1.06,   1.18], 
            [0.30,   0.54,   0.82,   0.98,   1.08], 
            ]
AA_uni[3,4] = np.copy(AA_uni_flop[3,4])
"""
2% strain - tr 0.5
"""
M_uni[3,5] = [ 
            [0.54,   0.92,   1.40,   1.70,   1.94], 
            [0.54,   0.92,   1.38,   1.68,   1.92], 
            [0.52,   0.90,   1.36,   1.66,   1.90], 
            [0.52,   0.90,   1.36,   1.66,   1.90], 
            [0.46,   0.84,   1.28,   1.58,   1.80], 
            ]
AA_uni_flip[3,5] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni_flop[3,5] = [ [nnnn for i in range(5)] for i in range(5)]
AA_uni[3,5] = [ [nnnn for i in range(5)] for i in range(5)]
"""
1% strain - tr 0
"""
M_uni[4,0] = [ 
            [0.54,   0.94,   1.42,   1.72,   1.98], 
            [0.54,   0.94,   1.42,   1.72,   1.94],
            [0.54,   0.94,   1.40,   1.72,   1.94], 
            [0.54,   0.94,   1.40,   1.72,   1.96],
            [0.52,   0.92,   1.38,   1.68,   1.94],
            ]
AA_uni_flip[4,0] = [   
            [nnnn,   nnnn,   0.16,   0.14,   0.12], 
            [nnnn,   nnnn,   0.16,   0.14,   0.12],
            [nnnn,   nnnn,   0.16,   0.14,   0.12],
            [nnnn,   nnnn,   0.16,   0.14,   0.12],
            [nnnn,   nnnn,   0.14,   0.12,   0.12], 
            ]
AA_uni_flop[4,0] = [   
            [0.14,   0.18,   nnnn,   nnnn,   nnnn], 
            [0.14,   0.18,   nnnn,   nnnn,   nnnn],
            [0.14,   0.18,   nnnn,   nnnn,   nnnn],
            [0.12,   0.18,   nnnn,   nnnn,   nnnn],
            [0.10,   0.14,   nnnn,   nnnn,   nnnn],                 
            ]
AA_uni[4,0] = [        
            [0.14,   0.18,   0.16,   0.14,   0.12], 
            [0.14,   0.18,   0.16,   0.14,   0.12],
            [0.14,   0.18,   0.16,   0.14,   0.12],
            [0.12,   0.18,   0.16,   0.14,   0.12],
            [0.10,   0.14,   0.14,   0.12,   0.12], 
            ]
"""
0.5% strain - tr 0
"""
M_uni[5,0] = [ 
            [0.54,   0.94,   1.42,   1.72,   1.98], 
            [0.54,   0.94,   1.42,   1.72,   1.98], 
            [0.54,   0.94,   1.42,   1.72,   1.96], 
            [0.54,   0.94,   1.42,   1.72,   1.96], 
            [0.54,   0.94,   1.40,   1.72,   1.96], 
            ]
AA_uni_flip[5,0] = [   
            [nnnn,   nnnn,   0.16,   0.14,   0.12], 
            [nnnn,   nnnn,   0.16,   0.14,   0.12], 
            [nnnn,   nnnn,   0.16,   0.14,   0.12], 
            [nnnn,   nnnn,   0.16,   0.14,   0.12], 
            [nnnn,   nnnn,   0.16,   0.14,   0.12], 
            ]
AA_uni_flop[5,0] = [   
            [0.14,   0.18,   nnnn,   nnnn,   nnnn], 
            [0.14,   0.18,   nnnn,   nnnn,   nnnn], 
            [0.14,   0.18,   nnnn,   nnnn,   nnnn], 
            [0.14,   0.18,   nnnn,   nnnn,   nnnn], 
            [0.14,   0.18,   nnnn,   nnnn,   nnnn], 
            ]
AA_uni[5,0] = [        
            [0.14,   0.18,   0.16,   0.14,   0.12], 
            [0.14,   0.18,   0.16,   0.14,   0.12], 
            [0.14,   0.18,   0.16,   0.14,   0.12], 
            [0.14,   0.18,   0.16,   0.14,   0.12], 
            [0.14,   0.18,   0.16,   0.14,   0.12], 
            ]





