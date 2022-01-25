##Working space for EOF simulation
#Last checked 20/01


import numpy as np
import matplotlib.pyplot as plt
import EOF_module_simu as mod
import math

#Simulation of EOF based on the module EOF_module_simu
#History vectors are created based on translated, noisy real data 
#In the implemented version, we won't have validation but just training
#It also won't be necessary to include noise characteristics as we don't know them
#The history vector will be a parameter of the functions and not created inside

l = 120 #Number of history vectors => number of time samples is l = 40
m = 20  #Number of measurements in history vectors (number of captors) => We have m = 4 degrees of freedom
n = 10  #Number of time samples in history vectors => the temporal order of the filter is n = 5
temporal_lag = 3 #Delay between command and measurement
sigma_train = 0.05 #Noise variance
sigma_test = 0.05

def linear(x):
    return 0.2*x + 12
    
def quadratic(x):
    return 4*x**2 + 2*x + 12

def sinus(x): #Frequency is not equal to one
    return np.sin(x/(10*np.pi))

def DL(x,two_p=60): #only works when x ≃ 0
    """ Taylor Development of the sinus function in 0 at the order two_p)"""
    if (two_p%2 == 0) :
        p = int(two_p/2)
    else  :
        p = int((two_p-1)/2)
    a = 0
    for i in range(p):
        a += ( (-1)**i / math.factorial(2*i+1) ) * x**(2*i+1) 
    return a

#   return x-x**3/6+x**5/124-x**7/(7*6*5*4*3*2)+x**9/(9*8*7*6*5*4*3*2)

function_train = sinus
function_valid = sinus

#function_train = linear
#function_valid = linear


## Training step of the filter with all of the data
true_data,history,a_posteriori_matrix,filter_full = mod.filter_training(l,m,n,temporal_lag,sigma_train,function_train)

##Prediction vector of all m=4 values at time t
predicted_vector = filter_full.dot(history[:])
        
##Predicted value versus real value for the first sensor, training data
predicted_first_sensor = predicted_vector[0,:] #a_posteriori_matrix[0,:].dot( product ).dot( history[:] )
        
#Accounting for temporal lag
#a = predicted_first_sensor
a = np.roll(predicted_first_sensor, -temporal_lag, axis=0)


#Plot the prediction for training data
plt.plot(np.arange(l), [val[0] for val in true_data[:l]],'r+-', label = 'True position', linewidth=.5)
plt.plot(np.arange(l), a[0:l], 'b+-',label = 'Estimated position', linewidth=.5)
plt.plot(np.arange(l), a_posteriori_matrix[0,:], 'g+-',label = 'Measured value, {} times frames time lag'.format(temporal_lag), linewidth=.5)
plt.ylabel('Value of the first sensor [Ø]')
plt.xlabel('Temporal frame [dt]')
plt.title('Predicted vs real value for the first sensor, training data')
plt.legend(bbox_to_anchor = (1, 1), loc = 'upper right', prop = {'size': 8})
plt.show()




## Extrapolation of the new data using the trained filter
test_data,history_test,a_posteriori_matrix_test = mod.filter_validation(l,m,n,temporal_lag,sigma_test,function_valid)

##Predicted value versus real value for the first sensor, test data
predicted_vector_test = filter_full.dot( history_test[:] )
    
predicted_first_sensor_test = predicted_vector_test[0,:]
#b = predicted_first_sensor_test
    
b = np.roll(predicted_first_sensor_test, -temporal_lag, axis=0)


#Plot the prediction for test data
plt.plot(np.arange(l), [val[0] for val in test_data[:l]],'r+-', label = 'True position', linewidth=.5)
plt.plot(np.arange(l), b[0:l], 'b+-',label = 'Estimated position', linewidth=.5)
plt.plot(np.arange(l), a_posteriori_matrix_test[0,:], 'g+-',label = 'Measured value, {} times frames time lag'.format(temporal_lag), linewidth=.5)
plt.ylabel('Value of the first sensor [Ø]')
plt.xlabel('Temporal frame [dt]')
plt.title('Predicted vs real value for the first sensor, test data')
plt.legend(bbox_to_anchor = (1, 1), loc = 'upper right', prop = {'size': 8})
plt.show()

c = [ [val[0] for val in test_data[:l]], [val[0] for val in true_data[:l]] ]
plt.plot(np.arange(l), c[0],'r+-', np.arange(l,2*l), c[1], linewidth=.5)