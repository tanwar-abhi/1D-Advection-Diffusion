import numpy as np

def fx(p,beta,h,M,Y,i):
    '''
     defining the flux term(f1)  at the nodel points for each element  
     dirichlet boundary conditions are Defined so at the 1st  node and last node, 
     bcoz of homegeneous boundary (i.e v(0) = 0 and v(l) = 0)
     
    Parameters
    ----------
    p : polynomial order
    beta : diffusion coeficient
    h : element length
    M : no. of elements
    Y : initial solution
    i : iterator loop

    Returns
    -------
    f1 : 

    '''
    
    
    if p == 1:
        
        f1 = np.zeros((2,1),dtype=float)
        
        if i==0:
            f1[0] = (beta/(2*h))*(- 1*Y[i+1,0])
            f1[1] = (beta/(2*h))*(2*Y[i+1,1] - 2*Y[i+1,0] + 1*Y[i+2,0]-Y[i,1]) 
        elif i == M-1:
            f1[0] = (beta/(2*h))*(-2*Y[i,1] + 2*Y[i,0] - 1*Y[i+1,0] + Y[i-1,1])
            f1[1] = (beta/(2*h))*(- Y[i,1])
        else:    
            f1[0] = (beta/(2*h))*(-2*Y[i,1] + 2*Y[i,0] - 1*Y[i+1,0]+ Y[i-1,1])
            f1[1] = (beta/(2*h))*(2*Y[i+1,1] -  2*Y[i+1,0] + 1*Y[i+2,0]- Y[i,1])
    
    if p == 2:
        
        f1 = np.zeros((3,1),dtype=float)
        
        if i==0:
            f1[0] = (beta/(2*h))*( - 2*Y[i,0] - Y[i,2]) 
            f1[1] = 0
            f1[2] = (beta/(2*h))*( -Y[i,0] - 2*Y[i,2] + 2*Y[i+1,0] + Y[i+1,2])
            
        elif i == M-1:
            f1[0] = (beta/(2*h))*(Y[i-1,0] + 2*Y[i-1,2] - 2*Y[i,0] - Y[i,2]) 
            f1[1] = 0
            f1[2] = (beta/(2*h))*( -Y[i,0] - 2*Y[i,2] )
            
        elif i == M-2:
            f1[0] = (beta/(2*h))*(Y[i-1,0] + 2*Y[i-1,2] - 2*Y[i,0] - Y[i,2]) 
            f1[1] = 0
            f1[2] = (beta/(2*h))*( -Y[i,0] - 2*Y[i,2] + 2*Y[i+1,0] + Y[i+1,2])
            
        else:    
            f1[0] = (beta/(2*h))*(Y[i-1,0] + 2*Y[i-1,2] - 2*Y[i,0] - Y[i,2]) 
            f1[1] = 0
            f1[2] = (beta/(2*h))*( -Y[i,0] - 2*Y[i,2] + 2*Y[i+1,0] + Y[i+1,2])
            
    return f1


