import numpy as np
import math

def lgwt(N, a, b):
    '''
    This script is for computing definite integrals using Legendre-Gauss 
    Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval
    [a,b] with truncation order N
    
    This function is coded From MATLAB code Written by Greg von Winckel - 02/25/2004
    
    Parameters
    ----------
    N : no. of quadrature points
    a : starting interval
    b : end interval

    Returns
    -------
    x : Quadtature points
    w : Quadrature weights
    '''
    N = N - 1
    N1 = N + 1 
    N2 = N + 2
    
    xu = np.linspace(-1,1,N1)
    
    #Initial guess
    y = np.cos(np.array([float(2*i+1) for i in range(0,N+1)])*(math.pi/(2*N+2))) + (0.27/N1)*np.sin(math.pi*xu*N/N2)
    
    #Legendre-Gauss Vandermonde Matrix
    L = np.zeros((N1,N2), dtype = float)
    
    #Derivative of LGVM
    Lp = np.zeros((N1,N2), dtype = float)
    
    # Compute the zeros of the N+1 Legendre Polynomial 
    # using the recursion relation and the Newton-Raphson method
    y0 = 2
    eps =  2.220446049250313e-16
    
    while max(abs(y-y0)) > eps:
        L[:,0] = 1
        Lp[:] = 0
        
        L[:,1] = y
        
        for k in range(1,N1+1):
            L[:,k] = ( ((2*k-1)*y) *L[:,k-1]-(k-1) * L[:,k-2] )/k
        
        Lp = N2*(L[:,N1-1] - y *L[:,N2-1])/(1-y**2)
        y0 = y
        y = y0 - L[:,N2-1]/Lp
    
    # Linear map from[-1,1] to [a,b]
    x = (a*(1-y) + b*(1+y))/2
    
    # Compute weights
    w = (b-a) / ((1-y**2)*Lp**2)*(N2/N1)**2
    
    return x, w

