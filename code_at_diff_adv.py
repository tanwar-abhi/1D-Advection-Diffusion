#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:56:40 2021

@author: abhishek
"""

# Global variables used accross all functions
#global endTime, length, M, n, p, h, delta_t

import numpy as np
import math
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
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
    
    #######################################################################
    # Simply copy paste from the "LAD_DG_mod.py" code, based on order1_2
    #######################################################################
    
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
        # U = 0.1 + 0.05*math.exp(-25 * (x-0.5)**2) ?????
        t = 0
        U = (0.025/math.sqrt(0.000625+0.02*t))*np.exp( (-1*((x-0.5-t)**2))/(0.00125+0.04*t) )
        
    # slight modification ??????
    U[(p+1)*(N-1)+p,0] = U[0,0]
    Uinit = U
    
    # plotting solution without interpolation over nodes- domain lies between 0 and 1
    x = np.linspace(0,1,N*(p+1))
    plt.plot(x,U,'b-')
    plt.xlabel('x')
    plt.ylabel('state')
    plt.title('Initial condition t = 0')
    
    # interior penalty- will change to BR2 in next code attempt if this converges 
    res = np.zeros((N*(p+1),1),float)
    
    # mass matrix
    mass = np.zeros((N*(p+1),N*(p+1)),float)
    
    # CFL number
    CFL = 0.05/(2*p+1)
    
    # tolerance
    tol = 1e-6
    
    # time period
    T = 0.05
    
    # time step
    dellt = CFL*(dellx**2)
    
    # iteration number
    iter = np.ceil(T/dellt)
    
    # stability factor for interior penalty method- found by experimentation
    # DO NOT EDIT
    if p==0:
        eta = 2
    else:
        if N==2:
            eta = 0.4*p**2/dellx
        elif N==4:
            eta = 0.2*p**2/dellx
        elif N==8:
            eta = 0.1*p**2/dellx
        elif N==16:
            eta = 0.06*p**2/dellx
    
    # error vector
    err = np.zeros((iter,1),float)
    
    # Populating mass matrix
    for k in range(1, N+1):
        for i in range(1, p+2):
            for j in range(1, p+2):
                for q in range(1, nq+1):
                    mass[(p+1)*(k-1)+i-1,(p+1)*(k-1)+j-1] = 
                    mass[]((p+1)*(k-1)+i,(p+1)*(k-1)+j)+(qw(q,1)*
                                                       basis(qp(q,1),p,i)*
                                                       basis(qp(q,1),p,j));
                
    # transforming to global domain
    mass=mass*dellx/2