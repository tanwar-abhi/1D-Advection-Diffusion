import numpy as np
import math

def solutionT0(p, n, h, M, endTime, length):
    '''
    This function initializes the solution vectors.
    Parameters
    ----------
    p : polynomial order
    n : total nodal points
    h : length of element
    M : number of elements
    endTime : final time
    length : length of domain

    Returns
    -------
    X : nodal coordinates of domain
    U0 : initail solution at t=0
    Uexact : exact solution
    '''
    
    if (p==1):
        X = np.linspace((0,0),(length,length),(n))      
        U0 = np.zeros((n,2), dtype = float)          

        # the 1st coloumn(left value at  the node)
        # tne 2nd coloumn (right value at the node)
        X[0,0] = 0
        X[0,1] = h
        for i in range(0 , M-1):
            if i == M-2:
                X[M-1,0] = X[i,0] + h
                X[M-1,1] = X[i,1] + h
            else:
                X[i+1,0] = X[i,0] + h
                X[i+1,1] = X[i,1] + h      
               
    
    if (p==2):
        X = np.zeros((M,3), dtype = float)
        U0 = np.zeros((M,3), dtype = float)
        Uexact = np.zeros((M,3), dtype = float)
    
    # the 1st coloumn(left value for  the element)
    # tne 2nd coloumn (middle value for the element)    
    # tne 3rd coloumn (right value for the element)        
        X[0,0] = 0
        X[0,1] = h/2
        X[0,2] = h
    
        for i in range(0 , M-1):
            if i==M-2:
                X[M-1,0] = X[i,0] + h
                X[M-1,1] = X[i,1] + h
                X[M-1,2] = X[i,2] + h
            else:
                X[i+1,0] = X[i,0] + h
                X[i+1,1] = X[i,1] + h
                X[i+1,2] = X[i,2] + h
        
    return X, U0, Uexact
        

def Uinit(X,t):
    
    Uexact = (0.025/math.sqrt(0.000625+0.02*t))*np.exp((-1*((X-0.5-t)**2))/(0.00125+0.04*t))
  
    return Uexact

 
        
