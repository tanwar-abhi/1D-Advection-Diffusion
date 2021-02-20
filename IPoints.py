
import numpy as np

def IP(p,h,M,beta,D,i):
    '''
    Integratioin points

    Parameters
    ----------
    p : polynomial order
    h : length of element
    M : no. of elements
    beta : diffusion coefficient
    D : TYPE
    i : element loop iterator

    Returns
    -------
    beta*f2 : 

    '''
    
    
    if p == 1:    
        
        f2 = np.zeros((2,1),dtype=float)
        
        if i == 0:
            
            f2[0] =  0 
            f2[1] = (1/(2*h))*( 2*D[i+1,0] - 2*D[i+1,1])
             
        elif i == M-1:
            f2[0] = (1/(2*h))*(-2*D[i,0] + 2*D[i,1])   
            f2[1] = 0
      
        else:
            f2[0] = (1/(2*h))*(-2*D[i,0] + 2*D[i,1])   
            f2[1] = (1/(2*h))*( 2*D[i+1,0] - 2*D[i+1,1])
            
            
    elif p == 2:
        
        f2 = np.zeros((3,1),dtype=float)
        
        if i == 0:      
            f2[0] = (1/(2*h))*(2*D[i,0] ) 
            f2[1] = 0
            f2[2] = (1/(2*h))*( -2*D[i+1,0] + 2*D[i,2])
           
        elif i == M-1:       
            f2[0] = (1/(2*h))*(2*D[i,0] - 2*D[i-1,2]) 
            f2[1] = 0
            f2[2] = (1/(2*h))*(  2*D[i,2])
            
        elif i ==M-2:
            f2[0] = (1/(2*h))*(2*D[i,0] - 2*D[i-1,2]) 
            f2[1] = 0
            f2[2] = (1/(2*h))*( -2*D[i+1,0] + 2*D[i,2])  
                
        else:
            f2[0] = (1/(2*h))*(2*D[i,0] - 2*D[i-1,2]) 
            f2[1] = 0
            f2[2] = (1/(2*h))*( -2*D[i+1,0] + 2*D[i,2])        
                

    return beta*f2
    