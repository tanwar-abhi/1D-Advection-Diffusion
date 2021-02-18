

def fn(xi, p, m):
    '''
    Parameters
    ----------
    xi : Quadrature points coordinates
    p : polynomial order
    m : element loop index

    Returns
    -------
    val : function value
    '''
    val = 1
    if p != 0:
        xij = -1 + (2*(m)/p)
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



def Grad(xi, p, m):
    '''
    This function calculates the gradient of the basis function.
    
    Parameters
    ----------
    xi : Quadrature points coordinates
    p : polynomial order
    m : element loop index
    
    Returns
    -------
    grad_val : value of gradient
    '''
    grad_val = 0.0
    
    if p != 0:
        xij = -1 + (2*m/p)
    else:
        xij = 1
    
    for g in range(1,p+2):
        if p != 0:
            xig = -1 + (2*(g-1)/p)
        else:
            xig = 1
            
        prod = 1
        
        if xij != xig:
            for k in range(1,p+2):
                if p != 0:
                    xik = -1 + (2*(k-1)/p)
                else:
                    xik = 1
                
                if xij != xik and xik != xig:
                    prod *= (xi - xik) / (xij - xik)
            grad_val += (prod/(xij-xig));
            
            
    return grad_val