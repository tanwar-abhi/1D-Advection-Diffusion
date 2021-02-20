#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:56:40 2021

@author: abhishek
"""

import elemental_matrix, initialize, quadrature, basis, function
import residual
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

    # Exact solution
    Uexact = initialize.Uinit(X,endTime)
    
    # initial value at time t =0
    U0 = initialize.Uinit(X, 0)
    
    # Defining the time loop and inside the time loop the elemental loop 
    # Time stepping algorithm  = Runge Kutta 3rd order( 3 step) 
    t = 0
    
    if p == 1:
        
        # initializing the final solution variable(U).
        uOut = np.zeros((n,2),dtype=float)
        
        # time loop
        while (t<endTime):
            print("TIMELOOP:-",t)    
            
            # elemental loop
            for i in range(0,N):
                # for each element 2 nodes will be there for 1st order
                u0 = np.zeros((2,1),dtype=float)      
                
                # for left node of an element the right value of the node is that needs to find out
                u0[0] = U0[i,1]
                
                # for right  node of an element the left value of the node is that needs to find out 
                u0[1] = U0[i+1,0]
                
                #Runge Kutta 1st step
                Utemp1 = u0 + delta_t*residual.resid(p,beta,h,N,U0,i,u0,K_diffusion, K_advection, mass_matrix)
                
                #Runge Kutta 2nd step  
                Utemp2 = 0.75*u0 + 0.25*Utemp1 +0.25*delta_t*residual.resid(p,beta,h,N,U0,i,Utemp1,K_diffusion, K_advection, mass_matrix) 
           
                # Runge Kutta 3rd step
                #resid(p,beta,h,N,U0,i,u0, K_diffusion, K_advection, mass_matrix):
                Utemp3 = (1/3)*u0 +(2/3)*Utemp2 + (2/3)*delta_t*residual.resid(p,beta,h,N,U0,i,Utemp2,K_diffusion, K_advection, mass_matrix)
                
                uOut[i,1] = Utemp3[0]                   
                uOut[i+1,0] = Utemp3[1]
          
                Up = np.zeros((2*n,1),dtype=float)        
                Xp = np.zeros((2*n,1),dtype=float)
                
                for i in range(0,n):
                    Up[2*i,0]=uOut[i,0]
                    Up[2*i+1,0]=uOut[i,1]
                    Xp[2*i,0]=X[i,0]
                    Xp[2*i+1,0]=X[i,1]
                    
                    
            #once all the element have been solved then the U0(initial solution) will be updated by Uout(final solution after 1st time iteration)
            U0 = uOut
            # increase the time step by delta_t
            t += delta_t 
            
    elif p == 2:
        uOut = np.zeros((N,3),dtype=float)
        while (t<endTime):
            print("TIMELOOP:-",t)    
            for i in range(0,N):
                #print("element-",i)
                u0 = np.zeros((3,1),dtype=float)
         
                u0[0] = U0[i,0]
                u0[1] = U0[i,1] 
                u0[2] = U0[i,2]
           
          
                Utemp1 = u0 + delta_t*residual.resid(p,beta,h,N,U0,i,u0,K_diffusion, K_advection, mass_matrix)
                Utemp2 = 0.75*u0 + 0.25*Utemp1 +0.25*delta_t*residual.resid(p,beta,h,N,U0,i,Utemp1,K_diffusion, K_advection, mass_matrix)
                Utemp3 = (1/3)*u0 +(2/3)*Utemp2 + (2/3)*delta_t*residual.resid(p,beta,h,N,U0,i,Utemp2,K_diffusion, K_advection, mass_matrix)
            
            #Utemp3 is the final vector after applying Runge Kutta and will have two values 
            # the 1st  value of the Utemp3 for  each element will be the left value  of that element        
            # the 2nd  value of the Utemp3 for  each element will be the  middle value  of that element      
            # the 3rd  value of the Utemp3 for  each element will be the right value  of that element           
            
                uOut[i,0]=Utemp3[0]
                uOut[i,1]=Utemp3[1]
                uOut[i,2]=Utemp3[2]
        
                Up = np.zeros((3*N,1),dtype=float)
                Xp = np.zeros((3*N,1),dtype=float)
            
                for i in range(0,N):
                    Up[3*i,0] = uOut[i,0]
                    Up[3*i+1,0] = uOut[i,1]
                    Up[3*i+2,0] = uOut[i,2]
                    Xp[3*i,0] = X[i,0]
                    Xp[3*i+1,0] = X[i,1]
                    Xp[3*i+2,0] = X[i,2]
            U0 = uOut
            t += delta_t
            
      
    
    # Calculating the L2 norm
    L2_norm  = math.sqrt(h)*np.linalg.norm((Uexact-uOut),'fro')
    order_value=len(re.search('\d+\.(0*)', str(L2_norm)).group(1)) + 1
    
    # initial soln
    Uin = initialize.Uinit(X,0)
    
    plt.plot(X,Uin,color='red')
    plt.plot(Xp,Up,'green')   
    plt.xlabel('X')
    plt.ylabel('U')
    plt.title("Uinit and Uapprox")  

    plt.legend(['Initial'],['Approximate']); #not sure why legend makes init soln title appear 2-3 times
    plt.show(); 
    
     


    
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
            
    plt.clf()
    
    # main iteration loop
    for TIME in range(1, int(iterator)):
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
                Uinterp = speed * U[(p+1)*(k-1):(p+1)*(k-1)+p+1].transpose().dot(basis_mat)             
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
            Uinterp = U[(p+1)*(k-1):(p+1)*(k-1)+p+1,0].dot(basisG_mat)
            res[(p+1)*(k-1):(p+1)*(k-1)+p+1] += basisG_mat.dot(np.diag(Uinterp.flatten().tolist())).dot(qw*2/dellx)
            
            for n in range(p+1):
                uL = U[(p+1)*(k-1)+p,0]
                uR = U[(p+1)*(k-1)+p,0]
                uHAT = 0.5*(uL + uR)
                sigma = 0
                vL = 0
                
                for m in range(p+1):
                    vL += U[(p+1)*(k-1)+m ,0] * basis.Grad(1,p,m)
                
                sigma += (vL/dellx) - (eta*(uL-uR)/dellx)
                res[(p+1)*(k-1)+n,0] -= basis.fn(1,p,n)*sigma - (2*(uL-uHAT) * basis.Grad(1,p,n)/dellx)
                
            # addition of advection terms
            if adv == 1:
                # analytical flux at quadrature points
                Uinterp = speed * U[(p+1)*(k-1):(p+1)*(k-1)+p+1,0].dot(basis_mat)
                res[(p+1)*(k-1):(p+1)*(k-1)+p+1] -= basisG_mat.dot(np.diag(Uinterp.flatten().tolist()).dot(qw))
                
                # boundary term
                for n in range(p+1):
                    uLl = U[(p+1)*(k-1-1)+p,0]
                    uRl = U[(p+1)*(k-1),0]
                    uLr = U[(p+1)*(k-1)+p,0]
                    uRr = U[(p+1)*(k-1)+p,0]
                    res[(p+1)*(k-1)+n] += (basis.fn(1,p,n) * function.rusadv(uLr,uRr)) - (basis.fn(-1,p,n) * function.rusadv(uLl,uRl))
            
            # interior elements
            for k in range(2,N-1):
                # interpolated state gradients over quad points
                Uinterp = U[(p+1)*(k-1):(p+1)*(k-1)+p+1,0].dot(basisG_mat)                
                res[(p+1)*(k-1):(p+1)*(k-1)+p+1] += basisG_mat.dot(np.diag(Uinterp.flatten().tolist())).dot(qw*2/dellx)
                
                # advection interior integral
                if adv == 1:
                    # analytical flux at quadrature points
                    Uinterp = speed * U[(p+1)*(k-1):(p+1)*(k-1)+p+1,0].dot(basis_mat)
                    res[(p+1)*(k-1):(p+1)*(k-1)+p+1] -= basisG_mat.dot(np.diag(Uinterp.flatten().tolist())).dot(qw)
                
                # boundary integrals
                # state discrepancy
                for n in range(p+1):
                    stateL = U[(p+1)*(k-1)+p,0]
                    stateR = U[(p+1)*(k+1-1),0]
                    Uavg = 0.5*(stateL + stateR)
                    termL = stateL - Uavg
                    termR = stateR - Uavg
                    
                    res[(p+1)*(k-1)+n] -= termL * basis.Grad(1,p,n)*2/dellx
                    res[(p+1)*(k+1-1)+n] += termR * basis.Grad(-1,p,n)*2/dellx
                    
                    # gradient discrepancy
                    sigma = 0
                    vL = 0
                    vR = 0
                    
                    for m in range(p+1):
                        vL += U[(p+1)*(k-1)+m,0] * basis.Grad(1,p,m)
                        vR += U[(p+1)*(k)+m,0] * basis.Grad(-1,p,m)
                        
                    sigma += (vL/dellx) + (vR/dellx) - (eta/dellx) * (stateL - stateR)
                    
                    res[(p+1)*(k-1)+n] -= basis.fn(1,p,n) * sigma
                    res[(p+1)*(k)+n] += basis.fn(-1,p,n) * sigma
                    
                    # advection boundary terms
                    if adv == 1:
                        uLl = U[(p+1)*(k-1-1)+p, 0] 
                        uRl = U[(p+1)*(k-1), 0]
                        uLr = U[(p+1)*(k-1)+p, 0]
                        uRr = U[(p+1)*(k), 0] 
                        res[(p+1)*(k-1)+n] += (basis.fn(1,p,n) * function.rusadv(uLr,uRr)) - (basis.fn(-1,p,n) * function.rusadv(uLl,uRl))
            
            # Compute solution using Runge-Kutta method
            if rk == 0:
                F0 = -np.linalg.inv(mass).dot(res)
                U0 = U
                U -= (0.5 * dellt * np.linalg.inv(mass).dot(res))
                
            elif rk == 1:
                F1 = -np.linalg.inv(mass).dot(res)
                U -= (dellt * 0.5 * np.linalg.inv(mass).dot(res))
                
            elif rk == 2:
                F2 = -np.linalg.inv(mass).dot(res)
                U -= (dellt * np.linalg.inv(mass).dot(res))
                
            elif rk == 3:
                F3 = -np.linalg.inv(mass).dot(res)
                U = U0 + ( dellt * (F0 + 2*F1 + 2*F2 + F3)/6)
                
            
            '''
            plt.plot(np.linspace(0,1,N*(p+1)), U, 'b-')
            plt.xlabel('x')
            plt.ylabel('State')
            plt.title('State at t=t ')
            #time.sleep(0.001)
            '''
    
    #else plt.plot(x,U,'b-')
    #plt.plot(np.linspace(0,1,N*(p+1)), U, 'b-')
    plt.plot(x, U, 'b-')
    plt.xlabel('x')
    plt.ylabel('State')
    plt.title('State at t=t ')
    
    plt.plot(x,Uinit, 'r-')
    plt.legend(["U", "Uinit"], loc ="upper left")
    plt.show()
