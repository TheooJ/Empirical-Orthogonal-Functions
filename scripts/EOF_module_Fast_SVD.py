#EOF Guyon for implementation on HciPy
#USE sklearn.decomposition.TruncatedSVD
# min(F) ||FD - P||^2 + alpha||F||^2
#Created on 03/02 - USE 
#Last checked 05/02


import numpy as np
import matplotlib.pyplot as plt


## Implementation of the Empirical Orthogonal Functions from Guyon et al. 2017
## Compute the filter F from either a matrix of all concatenated history data
## or using a moving concatenated matrix of history data


##                Training step of the filter with all of the data

#l = 100 #Number of history vectors in data matrix
#m = 4  #Number of measurements in history vectors 
#n = 10  #Number of time samples in history vectors 
#temporal_lag = 3 #Delay between the phenomenom happening and acquisition
#window = 5 #Size of the window of the moving average
#
#alpha = 1e-3 # Tiknhonov regularization of the filter



def filter_training_full_data_trunc_SVD(l,m,n,alpha,data_matrix,a_posteriori_matrix):
    
    #Computing of the filter
    if alpha == 0:
        U, sing, VT = np.linalg.svd(data_matrix.T)
        epsilon = np.zeros((l, n*m), dtype=complex) 
        epsilon[:min(n*m,l), :min(n*m,l)] = np.diag(sing)
        
        epsilon_pseudo_invert = np.linalg.pinv(epsilon)
    
        product = U.dot( epsilon_pseudo_invert.T ).dot( VT )
        filter_full = a_posteriori_matrix.dot( product )    
        
    else:
        data_matrix_reg = np.c_[ data_matrix, np.sqrt(alpha)*np.eye(n*m) ]
        a_posteriori_matrix_reg = np.c_[ a_posteriori_matrix, np.zeros([m,n*m]) ]
        
        U, sing, VT = np.linalg.svd(data_matrix_reg.T)
        epsilon = np.zeros((l+n*m, n*m), dtype=complex) 
        epsilon[:min(n*m,l+n*m), :min(n*m,l+n*m)] = np.diag(sing[:min(n*m,l+n*m)])
        
        epsilon_pseudo_invert = np.linalg.pinv(epsilon)
    
        product = U.dot( epsilon_pseudo_invert.T ).dot( VT )
        filter_full = a_posteriori_matrix_reg.dot( product )


    return filter_full



def filter_training_moving_average_trunc_SVD(l,m,n,alpha,window,data_matrix,history_new,a_posteriori_matrix,a_posteriori_new):
    """ function that returns the history vectors, the data matrix, 
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