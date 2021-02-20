#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:56:40 2021

@author: abhishek
"""

import elemental_matrix, initialize, quadrature, basis, function

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.sparse import diags
import re


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
    
    ##########################################################################
    # Simply copy paste rest from "LAD_DG_mod.py" code, based on order1_2    #
    ##########################################################################
    
else:
    speed = 4
    FS = 0                                    #free stream/free stream preservation test indicator
    adv = 1                                   #advection on/off
    x = np.zeros((N*(p+1),1),float)
    U = np.zeros((N*(p+1),1),float)
    dellx = 1/N                               #length of each element
    
    # Quadrature points
    nq = p + 5                              #no. of quadrature points
    qp,qw = quadrature.lgwt(nq,-1,1)
    qp[:,0].sort()
    
    for i in range(1,N+1):
        # global location of first node of element
        xbeg = dellx * (i-1)
        
        for j in range(1,p+2):
            node = (p+1) * (i-1) + j
            if p==0:
                x[node-1,0] = xbeg
            else:
                x[node-1,0] = xbeg + ((j-1)*dellx/p)
        
        
    if FS == 1:
        U[:,0] = 0.1
    else:
        # U = 0.1 + 0.05*math.exp(-25 * (x-0.5)**2) ?????
        t = 0
        U = (0.025/math.sqrt(0.000625+0.02*t))*np.exp( (-1*((x-0.5-t)**2))/(0.00125+0.04*t) )
        
    # slight modification ??????
    U[(p+1)*(N-1)+p,0] = U[0,0]
    Uinit = U
    
    # plotting solution without interpolation over nodes- domain lies between 0 and 1
    # x = np.linspace(0,1,N*(p+1))
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
    iterator = np.ceil(T/dellt)
    
    # stability factor for interior penalty method- found by experimentation
    # DO NOT EDIT
    if p == 0:
        eta = 2
    else:
        if N == 2:
            eta = 0.4*p**2/dellx
        elif N == 4:
            eta = 0.2*p**2/dellx
        elif N == 8:
            eta = 0.1*p**2/dellx
        elif N == 16:
            eta = 0.06*p**2/dellx
    
    # error vector
    err = np.zeros((int(iterator),1),float)

    # Populating mass matrix
    for k in range(N):
        for i in range(p+1):
            for j in range(p+1):
                for q in range(nq):
                    mass[(p+1)*(k-1)+i,(p+1)*(k-1)+j] += qw[q,0] * basis.fn(qp[q,0],p,i) * basis.fn(qp[q,0],p,j)
                
    # transforming to global domain
    mass = mass*dellx/2
    
    # matrix of basis function values at each of the quad points for ease of state evaluation
    basis_mat = np.zeros((p+1,nq), float)
    basisG_mat = np.zeros((p+1,nq), float)
    
    for i in range(p+1):
        for q in range(nq):
            basis_mat[i,q] = basis.fn(qp[q,0],p,i)
            basisG_mat[i,q] = basis.Grad(qp[q,0],p,i)
            
    # main iteration loop
    for time in range(1, int(iterator)):
        for rk in range(4):
            res *= 0
            
            k = 1
            Uinterp = np.dot(np.transpose(U[(p+1)*(k-1):(p+1)*(k-1)+p+1]), basisG_mat)
            res[(p+1)*(k-1):(p+1)*(k-1)+p+1] += basisG_mat.dot(np.diag(Uinterp.flatten().tolist()) ).dot(qw*2/dellx)
            
            for n in range(p+1):
                stateL = U[(p+1)*(k-1)+p,0]
                stateR = U[(p+1)*k,0]
                Uavg = 0.5*(stateL + stateR)
                termL = stateL - Uavg
                termR = stateR - Uavg
                res[(p+1)*(k-1)+n,0] -= termL * basis.Grad(1,p,n) * 2/dellx
                res[(p+1)*k+n,0] += termR * basis.Grad(-1,p,n) * 2/dellx
                
                # gradient discrepancy
                sigma = 0
                vL = 0
                vR = 0
                
                for m in range(p+1):
                    vL += U[(p+1)*(k-1)+m,0] * basis.Grad(1,p,m)
                    vR += U[(p+1)*k+m,0] * basis.Grad(-1,p,m)
                
                # convert array to float
                #vL = vL[0]
                #vR = vR[0]
                
                sigma += (vL/dellx) + (vR/dellx) - (eta*(stateL - stateR)/dellx) 
                res[(p+1)*(k-1)+n,0] -= basis.fn(1,p,n) * sigma
                res[(p+1)*k+n,0] += basis.fn(-1,p,n) * sigma
                
                # between left boundary and elem 1- EDIT
                uL = U[0,0]
                uR = U[(p+1)*(k-1),0]
                uHAT = 0.5 * (uL + uR)
                sigma = 0
                vR = 0
                
                for m in range(p+1):
                    vR += U[(p+1)*(k-1)+m,0] * basis.Grad(-1,p,m)
                    
                sigma += (vR/dellx) - (eta*(uL-uR)/dellx)
                res[(p+1)*(k-1)+n] += (basis.fn(-1,p,n) * sigma) + (2*(uR-uHAT)*basis.Grad(-1,p,n)/dellx)
            
            # addition of advection terms
            if adv == 1:
                # analytical flux at quadrature points
                #Uinterp = speed * U[(p+1)*(k-1):(p+1)*(k-1)+p+1] * basis_mat
                Uinterp = speed * U[(p+1)*(k-1):(p+1)*(k-1)+p+1].transpose().dot(basis_mat)
                
                #res[(p+1)*(k-1):(p+1)*(k-1)+p+1,0] -= basisG_mat * np.diag(Uinterp) * qw
                res[(p+1)*(k-1):(p+1)*(k-1)+p+1] -= basisG_mat.dot(np.diag(Uinterp.flatten().tolist())).dot(qw)
                
                # boundary term
                for n in range(p+1):
                    uLl = U[(p+1)*(k-1),0]
                    uRl = U[(p+1)*(k-1),0]
                    uLr = U[(p+1)*(k-1)+p,0]
                    uRr = U[(p+1)*k,0]
                    res[(p+1)*(k-1)+n] += (basis.fn(1,p,n) * function.rusadv(uLr,uRr)) - (basis.fn(-1,p,n) * function.rusadv(uLl,uRl))
            
            # last element
            k = N
            Uinterp = U[(p+1)*(k-1):(p+1)*(k-1)+p+1] * basisG_mat
            res[(p+1)*(k-1):(p+1)*(k-1)+p+1] += basisG_mat * np.diag(Uinterp)*qw*2/dellx
            
            for n in range(p+1):
                uL = U[(p+1)*(k-1)+p]
                uR = U[(p+1)*(k-1)+p]
            
            
            
            
            
# Line 188 matlab
    
    
    
    
    
    
    
    
    