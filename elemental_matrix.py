import numpy as np
import sys
def ElemMatrix(p, beta, alpha, h):
    '''
    Parameters
    ----------
    p : polynomial order
    beta : diffuction coefficient
    alpha : advection coefficient
    h : length of each element

    Returns
    -------
    K_diffusion : : diffusion stiffness matrix
    K_advection : advection stiffness matrix
    mass_matrix : mass matrix
    '''
    
    if (p==1):
        K_diffusion = (beta/h)*np.array([[1,-1],[-1,1]])
        K_advection = (alpha/2)*np.array([[-1,1],[-1,1]])
        mass_matrix = (h/6)*np.array([[2,1],[1,2]])
        
    elif (p==2):

     K_diffusion = (beta/(3*h))*np.array([[7,-8,1],[-8,16,-8],[1,-8,7]])
     K_advection = (alpha/6)*np.array([[-3,4,-1],[-4,0,4],[1,-4,3]])
     mass_matrix = (h/30)*np.array([[4,2,-1],[2,16,2],[-1,2,4]])
     
    else:
        print('the polynomial order is out of range');
        sys.exit(1)
     
    return K_diffusion, K_advection, mass_matrix