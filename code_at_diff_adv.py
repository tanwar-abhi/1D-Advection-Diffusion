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
import elemental_matrix, initialize, quadrature

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
    dellx = 1/N                               #length of each element
    
    # Quadrature points
    nq = p + 5                              #no. of quadrature points
    qp,qw = quadrature.lgwt(nq,-1,1)
    qp = np.sort(qp)
    
    for i in range(1,N+1):
        # global location of first node of elemen
        xbeg = dellx * (i-1)
        
        for j in range(1,p+2):
            node = (p+1) * (i-1) + j
            if p==0:
                x[node-1,0] = xbeg
            else:
                x[node-1,0] = xbeg + ((j-1)*dellx/p)
        
    
        if FS==1:
        U[:,0] = 0.1
    else:
        U = 0.1 + 0.05*math.exp(-25 * (x-0.5)**2)
        t = 0
    
    
    
