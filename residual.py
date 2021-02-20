import numpy as np
import flux, IPoints


def resid(p,beta,h,N,U0,i,u0, K_diffusion, K_advection, mass_matrix):
    '''
    Defining the residual by considering all the fluxes and stiffness 
    vector on the rhs side and multiplying  by the inverse of the matrxix         

     Parameters
     ----------
     U0 : initial solution
     i : loop iterator
     u0 : element nodal solution
     N : no. of elements

     Returns
     -------
     residual : solution residual
     '''
     
    f1 = flux.fx(p,beta,h,N,U0,i) #fx(p,beta,h,M,Y,i)
    f2 = IPoints.IP(p,h,N,beta,U0,i) #IP(p,h,M,beta,D,i):
   
    K1 = np.dot(K_diffusion,u0)                 # Converting the diffusion matrix into a vecto form
    K2 = np.dot(K_advection,u0)                 # Converting the advection matrix into a vector form 
    summation = -K1 - K2 + f1 -f2                # Summing up all the terms
    residual = np.dot(np.linalg.inv(mass_matrix), summation)         # Multiplying the inverse with the summation to get the residual  
    
    return residual
