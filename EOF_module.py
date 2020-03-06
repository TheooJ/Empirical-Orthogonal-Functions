import numpy as np


"""Module for Empirical Orthogonal Functions.

Functions are:
    filter_training: compute the filter with real data , return the filter
"""



def filter_training(l,m,n,alpha,data_matrix,a_posteriori_matrix,data_type='complex'):
    """Function to compute filter using EOF.
    
    :param l: Number of history vectors in data matrix
    :param m: Number of measurements in history vectors 
    :param n: Number of time samples in history vectors 
    :param delta_t: Delay between the phenomenom happening and acquisition
    :param alpha: Tikhonov regularization of the filter
    :param data_type: 
        
    Return the filter to estimate every element in history vector
    """
    
    if data_type == 'complex': 
    #Compute filter with complex data
    
        if alpha == 0:
            U, sing, VT = np.linalg.svd(data_matrix.conj().T)
            epsilon = np.zeros((l, n*m), dtype=complex) 
            epsilon[:min(n*m,l), :min(n*m,l)] = np.diag(sing)
            
            epsilon_pseudo_invert = np.linalg.pinv(epsilon)
        
            product = U.dot( epsilon_pseudo_invert.conj().T ).dot( VT )
            filter_full = a_posteriori_matrix.dot( product )    
            
        else:
            data_matrix_reg = np.c_[ data_matrix, np.sqrt(alpha)*np.eye(n*m) ]
            a_posteriori_matrix_reg = np.c_[ a_posteriori_matrix, np.zeros([m,n*m]) ]
            
            U, sing, VT = np.linalg.svd(data_matrix_reg.conj().T)
            epsilon = np.zeros((l+n*m, n*m), dtype=complex) 
            epsilon[:min(n*m,l+n*m), :min(n*m,l+n*m)] = np.diag(sing[:min(n*m,l+n*m)])
            
            epsilon_pseudo_invert = np.linalg.pinv(epsilon)
        
            product = U.dot( epsilon_pseudo_invert.conj().T ).dot( VT )
            filter_full = a_posteriori_matrix_reg.dot( product )
    
    
        return filter_full
        
    elif data_type == 'real':
    #Compute filter with real data
        
        if alpha == 0:
            U, sing, VT = np.linalg.svd(data_matrix.T)
            epsilon = np.zeros((l, n*m)) 
            epsilon[:min(n*m,l), :min(n*m,l)] = np.diag(sing)
            
            epsilon_pseudo_invert = np.linalg.pinv(epsilon)
        
            product = U.dot( epsilon_pseudo_invert.T ).dot( VT )
            filter_full = a_posteriori_matrix.dot( product )    
            
        else:
            data_matrix_reg = np.c_[ data_matrix, np.sqrt(alpha)*np.eye(n*m) ]
            a_posteriori_matrix_reg = np.c_[ a_posteriori_matrix, np.zeros([m,n*m]) ]
            
            U, sing, VT = np.linalg.svd(data_matrix_reg.T)
            epsilon = np.zeros((l+n*m, n*m)) 
            epsilon[:min(n*m,l+n*m), :min(n*m,l+n*m)] = np.diag(sing[:min(n*m,l+n*m)])
            
            epsilon_pseudo_invert = np.linalg.pinv(epsilon)
        
            product = U.dot( epsilon_pseudo_invert.T ).dot( VT )
            filter_full = a_posteriori_matrix_reg.dot( product )
    
    
        return filter_full






#TO DO: "smart" moving average with rank-one update

def filter_training_moving_average(l,m,n,alpha,window,data_matrix,history_new,a_posteriori_matrix,a_posteriori_new):
    """ Function that returns the history vectors, the data matrix, 
    the a posteriori matrix and the full-order filter for the training set """
    
    ##Data matrix update with new history
    data_matrix = np.delete(data_matrix, [range(window)], axis=1)
    data_matrix = np.c_[history_new,data_matrix] #New history ON THE LEFT
            
            
    ##A posteriori measurement matrix update
    a_posteriori_matrix = np.delete(a_posteriori_matrix, [range(window)], axis=1) 
    a_posteriori_matrix = np.c_[a_posteriori_matrix,a_posteriori_new] 

    
    #Computing of the filter
    if alpha == 0:
        U, sing, VT = np.linalg.svd(data_matrix.T)
        epsilon = np.zeros((l, n*m), dtype=complex) 
        epsilon[:min(n*m,l), :min(n*m,l)] = np.diag(sing)
        
        epsilon_pseudo_invert = np.linalg.pinv(epsilon)
    
        product = U.dot( epsilon_pseudo_invert.T ).dot( VT )
        filter_MA = a_posteriori_matrix.dot( product )    
    else:
        data_matrix_reg = np.c_[ data_matrix, np.sqrt(alpha)*np.eye(n*m) ]
        a_posteriori_matrix_reg = np.c_[ a_posteriori_matrix, np.zeros([m,n*m]) ]
        
        U, sing, VT = np.linalg.svd(data_matrix_reg.T)
        epsilon = np.zeros((l+n*m, n*m), dtype=complex) 
        epsilon[:min(n*m,l+n*m), :min(n*m,l+n*m)] = np.diag(sing[:min(n*m,l+n*m)])
        
        epsilon_pseudo_invert = np.linalg.pinv(epsilon)
    
        product = U.dot( epsilon_pseudo_invert.T ).dot( VT )
        filter_MA = a_posteriori_matrix_reg.dot( product )


    
    return data_matrix,a_posteriori_matrix,filter_MA