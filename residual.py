import numps as np
import flux, IP


 def residual(U0,i,u0, K_diffusion, K_advection, mass_matrix):
     '''
    Defining the residual by considering all the fluxes and stiffness 
    vector on the rhs side and multiplying  by the inverse of the matrxix         

     Parameters
     ----------
     U0 : TYPE
         DESCRIPTION.
     i : TYPE
         DESCRIPTION.
     u0 : TYPE
         DESCRIPTION.
     mass_matrix : TYPE
         DESCRIPTION.

     Returns
     -------
     residual : TYPE
     '''
     
    f1 = flux.fx(U0,i)
    f2 = IP(U0,i)
   
    K1 = np.dot(K_diffusion,u0)                 # Converting the diffusion matrix into a vecto form
    K2 = np.dot(K_advection,u0)                 # Converting the advection matrix into a vector form 
    summation = -K1 - K2 + f1 -f2                # Summing up all the terms
    invM=np.linalg.inv(mass_matrix)           # Finding out the inverse 
    residual = np.dot(invM,summation)         # Multiplying the inverse with the summation to get the residual  
    
    return residual
