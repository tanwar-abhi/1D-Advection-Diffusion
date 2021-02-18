import numpy as np

def fn(xi, p, m):
    '''
    Parameters
    ----------
    xi : Quadrature points coordinates
    p : polynomial order
    m : loop index

    Returns
    -------
    val : TYPE
    '''
    val = 1
    if p != 0:
        xij = -1 + (2*(m-1)/p)
    else:
        xij = 1
        
    for k in range(1,p+2):
        if p!=0:
            xik = -1 + (2*(k-1)/p)
        else:
            xik = 1
        
        if xij != xik:
            val *= (xi - xik)/(xij - xik)
            
    # Reference element [-1, 1]
    return val