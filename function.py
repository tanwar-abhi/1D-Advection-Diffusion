
def rusadv(ul,ur):
    '''
    Parameters
    ----------
    ul : value at left node
    ur : value at right node

    Returns
    -------
    G : TYPE

    '''
    speed = 4
    Fl = speed * ul
    Fr = speed * ur
    
    G = 0.5*(Fl + Fr) - (0.5 * speed * (ur-ul))
    
    return G