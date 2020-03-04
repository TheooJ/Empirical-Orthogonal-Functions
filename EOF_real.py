"""Working space for EOF implementation with Blind Moving Average
Implementation of the Empirical Orthogonal Functions from Guyon et al. 2017
Compute the filter F from either a  moving matrix of concatenated history data

:param l: Number of history vectors in data matrix
:param m: Number of measurements in history vectors 
:param n: Number of time samples in history vectors 
:param delta_t: Delay between the phenomenom happening and acquisition
:param window: Size of the window of the moving average
:param alpha: Tiknhonov regularization of the filter
"""

#from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
import EOF_module as mod



#Initialization of the parameters
grid_size = 20

l = 5 #Nb of history vectors in data matrix
m = grid_size**2  #Nb of measurements in each time sample
n = 3  #Nb of time samples in a history vector
delta_t = 1 #Temporal position of the wavefront we want to estimate
window = 10 #Number of samples before recomputing the filter
            #Can be set to arbitrarily large value if we don't want a MA

alpha = 1e-3 # Tikhonov regularization parameter
if alpha < 0:
  raise Exception("Sorry, regularization parameter must be positive of null") 

#Initialization of the matrices 
  ## Uniform
E0 = np.ones([grid_size,grid_size])  #We don't have access to this in reality
data_matrix = np.zeros([m*n,l])
a_posteriori_matrix = np.zeros([m,l])

  ## Speckles
center_1 = [5,5]
center_2 = [15,15]
sigma_1 = 0.9
sigma_2 = 2.7

planet = np.zeros([grid_size,grid_size])
speckles = np.zeros([grid_size,grid_size])

for row in range(planet.shape[0]):
    for col in range(planet.shape[1]):
        planet[row][col] = np.exp( - np.sqrt( (row-center_1[0])**2 + (col-center_1[1])**2 )  / sigma_1**2)
        speckles[row][col] = np.exp( - np.sqrt( (row-center_2[0])**2 + (col-center_2[1])**2 )  / sigma_2**2)

camera = planet + speckles 

#Initialization of the lists
history_current = []
history_list = []
E_measured_list = []
E_hat_list = []
MSE_list = []


nb_hist = 0 
random_walk = np.zeros([grid_size,grid_size])
already_computed = False 
moving_average = 0

rms_diff_list=[]

#Estimation loop
for iteration in range(200):
    
    #Update of E
    random_walk += 0.05*np.random.normal(size=([grid_size,grid_size])) 
    #E = camera + random_walk + 0.1*np.random.poisson(abs(E0), size=([grid_size,grid_size]))#Drift
    E = camera + random_walk
   
    #E = E0 + 0.05*np.random.normal(size=([grid_size,grid_size])) #Normal gaussian noise
    
    #E = E0 + 0.03*np.random.poisson(abs(E0), size=([grid_size,grid_size]))

    
    #E = E0 + np.sin(iteration/(5*np.pi)) + 0.05*np.random.normal(size=([grid_size,grid_size]))
    #b =  0.03*np.random.normal(size=([grid_size,grid_size])) * np.roll(E0, -1, axis=1)
    #E = E0 + b
    
    E_measured_list.insert(0,E.copy())
    history_current = np.concatenate((E.copy().flatten(),history_current),axis=0)
        
    #Construction of history vectors
    if len(history_current) == m*n: #(>=)
        history_list.insert(0,history_current.copy())
        history_current = np.delete(history_current,range(len(history_current)-m,len(history_current)))
        nb_hist +=1
    
    #Check if we need to recompute the filter
    if moving_average == window: 
        already_computed = False
        moving_average = 0
    
    #Construction of the data matrix, a posteriori matrix and computing of the filter  
    if already_computed is False and nb_hist >= l+delta_t:
        for i in range(l):
            data_matrix[:,i] = history_list[i+delta_t]
            a_posteriori_matrix[:,i] = E_measured_list[i].flatten()
        F = mod.filter_training_real(l,m,n,alpha,data_matrix,a_posteriori_matrix)
        already_computed = True
    
    #Estimation of E if the filter has been computed
    if already_computed is True:
        E_hat = F.dot(history_list[0]).reshape(grid_size,grid_size)
        E_hat_list.insert(0,E_hat.copy())
        moving_average += 1



#Reorder the lists
E_hat_list.reverse() 
E_measured_list.reverse()


     ####Compute the MSE & Plot the prediction results (reordered lists)

#for i in range(delta_t,len(E_hat_list)):
#    plt.clf()
#    diff = E_hat_list[i] - E_measured_list[i+l+n-2+delta_t]
#    rms_diff = np.abs(diff)/np.abs(E_measured_list[i+l+n-2+delta_t])
#    squared_sum = sum(sum(abs(diff)**2)) #Must account for F??
#    MSE_list.append(squared_sum)
#
#    plt.subplot(131)
#    plt.imshow(abs(E_measured_list[i+l+n-2+delta_t]), cmap='inferno')
#    plt.title('Measured Field')
#    plt.colorbar()
#    
#    plt.subplot(132)
#    plt.imshow(abs(E_hat_list[i]), cmap='inferno')
#    plt.title('Estimated Field')
#    plt.colorbar()
#    
#    plt.subplot(133)
#    plt.imshow(rms_diff, vmin=0, vmax=0.1, cmap='inferno')
#    plt.title('Estimation Error')
#    plt.colorbar()
#    
#    plt.suptitle('Prediction Results, iteration {}'.format(i))
#    plt.pause(0.1)




####Plot the Estimation for the first pixel
plt.clf()
plt.plot(np.arange(l+n-2+delta_t,len(E_hat_list)+l+n-2+delta_t), [val[0][0] for val in E_hat_list], color='tab:orange', label = 'Estimated Electric Field', linewidth=1)
plt.plot(np.arange(len(E_measured_list)), [val[0][0] for val in E_measured_list], color='tab:blue', label = 'Measured Electric Field', linewidth=1)
plt.ylabel('Value of the first sensor [Ø]')
plt.xlabel('Iteration index k')
plt.title('Predicted versus measured value for the first sensor')
plt.legend(bbox_to_anchor = (1, 1), loc = 'upper right', prop = {'size': 10})
plt.show()
     
     



#Compute the MSE
if not(MSE_list):
    for i in range(delta_t,len(E_hat_list)):
        rms_diff = np.abs(E_hat_list[i] - E_measured_list[i+l+n-2+delta_t])#/np.abs(E_measured_list[i+l+n-2+delta_t])
        rms_diff_list.append(rms_diff) 
        squared_sum = sum(sum(abs(rms_diff)**2)) 
        MSE_list.append(squared_sum)




###Plot the MSE as a function of iterations
plt.clf()
plt.plot(np.arange(l+n-2+delta_t,len(MSE_list)+l+n-2+delta_t), [np.log(val) for val in MSE_list], color='tab:orange', label = 'log(MSE)', linewidth=1)
plt.plot(np.arange(len(E_measured_list)), [np.log(val[0][0]) for val in E_measured_list], color='tab:blue', label = 'log(Measured Electric Field)', linewidth=1)
plt.ylabel('log(MSE)')
plt.xlabel('Iteration index k')
plt.title('MSE versus measured value')
plt.legend(bbox_to_anchor = (1, 1), loc = 'upper right', prop = {'size': 10})
plt.show()

#
     #####Plot rmd_diff pour montrer recalcul du filtre
#for i in range(28,34):
#    plt.clf()
#    plt.imshow(rms_diff_list[i], vmin=0, vmax=0.1, cmap='inferno')
#    plt.title('Reconstruction MSE')
#    plt.colorbar()
#    plt.pause(0.1)
     
     
     
### Computing of the autocorrelation
autocorrelation_list = []
for i in range(len(history_list)):
    a = sum([x * np.conj(y) for x, y in zip(history_list[0], history_list[i])]) 
    autocorrelation_list.append(a.copy())
    
### Plot the autocorrelation
plt.clf()
plt.plot(np.arange(len(autocorrelation_list)), [val for val in autocorrelation_list], color='tab:blue', linewidth=1)
plt.ylabel('Autocorrelation [wf^2]')
plt.xlabel('Time shift')
plt.title('Autocorrelation of the history vector')
plt.show()

print('Autocorrelation slope is', (autocorrelation_list[-1]-autocorrelation_list[0])/len(autocorrelation_list))
