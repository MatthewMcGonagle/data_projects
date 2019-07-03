'''
Functions for getting periodic prediction features.
'''

import numpy as np

def make_periodic(x, periods, return_alphas = None):
    '''
    Get the periodic prediction features given time x and periods for the periodic features.
    Returns periodic features based on both sines and cosines.

    Parameters
    ----------
    x : Numpy array or Pandas Series
        The time features to turn into periodic features.
    periods : Numpy array or Panda Series
        The desired periods. Each period results in a pair of periodic sine and cosine features.
    return_alphas : None or String
        Which type of regularization alphas to return.
        'None' : Don't return any alphas
        'L1' : Regularization alphas depending on the L1 norm of the second derivatives.
        'L2' : Regularization alphas depending on the L2 norm of the second derivatives.

    Returns
    -------
    X_periodic : numpy.ndarray of shape (n_samples, 2 * len(periods))
        The periodic features.

    alphas : Numpy of shape (2 * len(periods),)
        When return_alphas != None, this is the regularization alphas associated with the
        periodic data.
    '''

    X_periodic = [[np.cos(x * 2 * np.pi / period), np.sin(x * 2 * np.pi / period)]
                 for period in periods]
    X_periodic = np.concatenate(X_periodic, axis = 0).T
    if return_alphas is None:
        return X_periodic
    
    if return_alphas == "L2":  
        alphas = [[ 1 / period**4, 1 / period**4] for period in periods]
        
    elif return_alphas == "L1":
        alphas = [[1 / period**2, 1 / period**2] for period in periods]
        
    else:
        raise Exception("Parameter return_alphas must be either 'L1', 'L2', or None.")
        
    alphas = np.array(alphas).reshape(-1)
    alphas /= np.linalg.norm(alphas)
    return X_periodic, alphas
