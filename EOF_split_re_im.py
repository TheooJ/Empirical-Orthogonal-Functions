##Working space for EOF implementation 
#One time computing of the filter with Tikhonov regularization and time lag
#Data is sampled on the fly and history is comprised of all known values (full m) 
#Created 23/01, corrects time scale errors in last file
#E is split into real and imaginary parts


#from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
import EOF_module as mod


#Initialization of the parameters
grid_size = 5

l = 40 #Nb of history vectors in data matrix
m = 2*grid_size**2  #Nb of measurements in each time sample
n = 5  #Nb of time samples in a history vector
delta_t = 1 #Temporal position of the wavefront we want to estimate

window = 3 #Can be set to arbitrarily large value if we don't want a MA


alpha = 1e-3 # Tikhonov regularization parameter
if alpha < 0:
  raise Exception("Sorry, no regularization parameters below zero") 



#Initialization of the matrices 
E = np.ones([grid_size,grid_size])*(1+1j) #We never have access to this 
data_matrix = np.zeros([m*n,l])
a_posteriori_matrix = np.zeros([m,l])

#Initialization of the lists
history_current = []
history_list = []
E_measured_list = []
E_real_hat_list = []
E_imag_hat_list = []
MSE_list = []

nb_hist = 0 
random_walk = np.zeros([grid_size,grid_size])
already_computed = False 
moving_average = 0



#Estimation loop
for iteration in range(200):
    
#    E_real += 0.05*np.random.normal(size=([grid_size,grid_size]))
#    E_imag += 0.05*np.random.normal(size=([grid_size,grid_size]))
    random_walk += 0.05*np.random.normal(size=([grid_size,grid_size]))
    E *= np.exp(1j * random_walk)
    E_measured_list.insert(0,E.copy())
    history_current = np.concatenate((E.real.copy().flatten(),E.imag.copy().flatten(),history_current),axis=0)
        
    #Construction of history vectors
    if len(history_current) >= m*n:
        history_list.insert(0,history_current.copy())
        history_current = np.delete(history_current,range(len(history_current)-m,len(history_current)))
        nb_hist +=1
    
    #Check if we need to recompute the filter
    if moving_average > window: 
        already_computed = False
        moving_average = 0
    
    #Construction of the data matrix, the a posteriori matrix and computing of the filter  
    if already_computed is False and nb_hist >= l+delta_t:
        for i in range(l):
            data_matrix[:,i] = history_list[i]
            a_posteriori_matrix[:,i] = np.concatenate( (E_measured_list[i+delta_t].real.flatten() \
                               , E_measured_list[i+delta_t].imag.flatten()) ).T
        F = mod.filter_training_full_data(l,m,n,alpha,data_matrix,a_posteriori_matrix)
        already_computed = True

    
    
    #Estimation of E if the filter has been computed
    if already_computed is True:
        E_hat = F.dot(history_list[0])
        E_real_hat = np.split(E_hat, 2)[0].reshape(grid_size,grid_size)
        E_imag_hat = np.split(E_hat, 2)[1].reshape(grid_size,grid_size)
        E_real_hat_list.insert(0, E_real_hat.copy() )
        E_imag_hat_list.insert(0, E_imag_hat.copy() )

    

##Plot the prediction results
#for i in range(len(E_real_hat_list)):
#    E_hat = E_real_hat_list[i] + E_imag_hat_list[i] 
#    diff = E_hat - E_measured_list[l+i]
#    squared_sum = sum(sum(abs(diff)**2)) #Careful because these are complex numbers
#    MSE_list.append(squared_sum)


#Compute the MSE & Plot the prediction results
for i in range(delta_t,len(E_real_hat_list)):
    plt.clf()
    diff = E_real_hat_list[i] + 1j * E_imag_hat_list[i]  - E_measured_list[i-delta_t]
    rms_diff = np.abs( diff )/np.abs(E_measured_list[i-delta_t])
    squared_sum = sum(sum(abs(diff)**2))
    MSE_list.append(squared_sum)
    
    plt.subplot(121)
    plt.imshow(rms_diff, vmin=0, vmax=1, cmap='inferno')
    plt.title('Estimation Error')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(abs(E_real_hat_list[i] + 1j * E_imag_hat_list[i]), cmap='inferno')
    plt.title('Estimated Field')
    plt.colorbar()
    plt.suptitle('Prediction Results')
    plt.pause(0.1)