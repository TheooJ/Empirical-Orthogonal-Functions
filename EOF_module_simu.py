#EOF Guyon for simulation
#Last checked 20/01

import numpy as np
import matplotlib.pyplot as plt

## Simulation experiments on the Empirical Orthogonal Functions from Guyon et al. 2017
## First we compute the filter F from a matrix of all concatenated history data
## When then compute it using a moving concatenated matrix of history data


###Améliorations:

#1) Créer une fonction et un module pour ne pas recalculer à chaque fois le
#vecteur historique dans le modèle MA (juste décalé)
#def filter_moving_average(l,m,n,temporal_lag,function,history_past):
   #pop dernier history
   #remplacer nouvelle value avec derniers measurements
      
#2) History as a parameter of filter_training pour pouvoir le décaler
#lors du moving average


##                Training step of the filter with all of the data

l = 100 #Number of history vectors => number of time samples is l = 40
m = 4  #Number of measurements in history vectors (number of captors) => We have m = 4 degrees of freedom
n = 10  #Number of time samples in history vectors => the temporal order of the filter is n = 5
temporal_lag = 3 #Delta_t, delay between the phenomenom happening and acquisition
sigma = 0.05


def filter_training(l,m,n,temporal_lag,sigma,function):
    """ function that returns the history vectors, the true data 
    the a posteriori matrix and the full-order filter for the training set"""
    
    ##True data: Perfect case (we don't have access to this)
    #Builds a list of arrays containing l*n samples of size m
    
    true_data = []
    
    for i in range(l*n):
        true_proxy = np.zeros(m,)
        for j in range(m):
            true_proxy[j] =  function(m*i+j)
        true_data.append(true_proxy)
             
        
    ##Measurements: Noisy measure of the true data
    #Suppose additive zero-mean, reduced gaussian noise
        
    measurements = [None]*( len(true_data) ) #(1,4)
    
    for i in range( len(measurements) ):
        for j in range(m):
            measurements[i] = true_data[i]+np.random.normal(size=(m,))*sigma #choice of sigma??   
    
    
    ##History vector: concatenation of n measurements
        
    history = [] #list of all n*m history vectors
                 #we take l of them to build D
    
    for i in range(l):
        hist_proxy = np.zeros((m*n,1))
        for j in range(n):
            for k in range(m):
                hist_proxy[m*j+k] = measurements[i+j][k] 
        history.append(hist_proxy)
            
        
    ##Data matrix: column matrix of all history vectors
    data_matrix = np.zeros((m*n,l))
    
    for i in range(l):
        data_matrix[:,i] = history[i].T #shape (20,) and (20,1)
     
    
    ##A posteriori measurement matrix for all captors
    #At each time, we must account for time lag: measurements are a noisy 
    #representation of the real value three steps ago 
    #delta_t = 3dt, no interpolation needed
    
    a_posteriori_matrix = np.zeros((m,l))
    
    for j in range(l):
        a_posteriori_matrix[:,j] = measurements[j-temporal_lag] #Takes into account the lag                                               
    a_posteriori_matrix[:,0:temporal_lag] = 0 #Set values from negative times  to zero (causality)
    
    
    ##Regressive filter for measurement estimation: training on all data
    #SVD of the Data matrix transpose
                                                     
    U, sing, V_transpose = np.linalg.svd(data_matrix.T)
    epsilon = np.zeros((l, n*m), dtype=complex) #Complex useless?
    epsilon[:min(n*m,l), :min(n*m,l)] = np.diag(sing)
    
    #Pseudo-invert of epsilon
    epsilon_pseudo_invert = np.linalg.pinv(epsilon)
    
    #Computing of the filter
    product = U.dot( epsilon_pseudo_invert.T ).dot( V_transpose )
    filter_full = a_posteriori_matrix.dot( product )
    
    
    
    return true_data,history,a_posteriori_matrix,filter_full



def filter_validation(l,m,n,temporal_lag,sigma,function):
    """ function that returns the history vectors, the true data 
    the a posteriori matrix for data after the training set"""
    
    ##                Extrapolation of the new data using the trained filter
    
    
    ##True test data taken after the training step
    
    
    test_data = [] #(m,)
    
    for i in range(l*n):
        test_proxy = np.zeros(m,)
        for j in range(m):
            #test_proxy[j] =  m*i+j 
            #test_proxy[j] =  np.sin(((l*n-1)+(m)*i+j)/(5*np.pi))
            test_proxy[j] = function(l*n-1+(m)*i+j)
        test_data.append(test_proxy)
        
        
    ##Measurements: Noisy measure of the test data
    #Suppose additive zero-mean, reduced gaussian noise
        
    measurements_test = [None]*( len(test_data) ) #(1,4)
    
    for i in range( len(measurements_test) ):
        for j in range(m):
            measurements_test[i]=test_data[i]+np.random.normal(size=(m,))*sigma #sigma??
    
    
    ##History vector: concatenation of n measurements of test data
        
    history_test = [] #list of all n*m history vectors
    
    for i in range(l):
        hist_test_proxy = np.zeros((m*n,1))
        for j in range(n):
            for k in range(m):
                hist_test_proxy[m*j+k] = measurements_test[i+j][k]
        history_test.append(hist_test_proxy)
       
        
#    ##Data matrix: column matrix of all history vectors
#    data_matrix_test = np.zeros((m*n,l))
#    
#    for i in range(l):
#        data_matrix_test[:,i] = history_test[i].T #shape (20,) and (20,1)
        #Not useful for validation
    
    
    ##A posteriori measurement matrix for all captors
    #At each time, we must account for time lag: measurements are a noisy 
    #representation of the real value three steps ago 
    #delta_t = 3dt, no interpolation needed
    
    a_posteriori_matrix_test = np.zeros((m,l))
    
    for j in range(l):
        a_posteriori_matrix_test[:,j] = measurements_test[j-temporal_lag] #Takes into account the lag                                               
    a_posteriori_matrix_test[:,0:temporal_lag]=0 #Set negative temporal values to zero (causality)
    
    
    
    return test_data,history_test,a_posteriori_matrix_test


##                Training step of the filter using a moving average
#The length of the window for the MA is window = 10 measurements



