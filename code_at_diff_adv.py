#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:56:40 2021

@author: abhishek
"""

# Global variables used accross all functions
#global endTime, length, M, n, p, h, delta_t

import numpy as np
from matplotlib import pyplot as plot
from scipy.sparse import diags
import re
import elemental_matrix, initialize

endTime = 0.05                                                  # final time               
alpha = 4                                                     # coefficient associated with advection term
beta= 1                                                      # coefficient associated with diffusion term
length = 1                                                    # length of the domain 
p = int(input("Enter the polynomial order either 0, 1, 2, 3 or 4: "))   #polynomial order
N = 8                                                   # no of elements
n =  p*N + 1                                                  # total no.of nodal points  
h = length/N                                                 # length of an element


if p==1 or p==2:
    cfl = 0.1/(2*p+1)                                 # cfl criteria to calculate the delta_t
    delta_t = (cfl*h*h)/beta                           # time step

    K_diffusion,K_advection,mass_matrix = elemental_matrix.ElemMatrix(p, beta, alpha, h)

    X, U0, Uexact = initialize.solutionT0(p, n, h, N, endTime, length)

    Uexact = initialize.Uinit(X,endTime)                   # Exact solution
    U0 = initialize.Uinit(X, 0)                            # initial value at time t =0
    
else:
    speed = 4
    FS = 0                                    #free stream/free stream preservation test indicator
    adv = 1                                   #%advection on/off
    x = np.zeros((N*(p+1),1),float)
    U = np.zeros((N*(p+1),1),float)
    dellx=1/N                               #length of each element