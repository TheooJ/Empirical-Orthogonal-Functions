##Working space for EOF implementation with "Blind" Moving Average
#Created 03/02: last version of EOF
#Last checked on 05/02


#from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
import EOF_module as mod



#Initialization of the parameters
grid_size = 5

l = 20 #Nb of history vectors in data matrix
m = grid_size**2  #Nb of measurements in each time sample
n = 10  #Nb of time samples in a history vector
delta_t = 1 #Temporal position of the wavefront we want to estimate
window = 4 #Can be set to arbitrarily large value if we don't want a MA

alpha = 1e-3 # Tikhonov regularization parameter
if alpha < 0:
  raise Exception("Sorry, regularization parameter must be positive of null") 

#Initialization of the matrices 
E0 = np.ones([grid_size,grid_size])  #We don't have access to this in reality
data_matrix = np.zeros([m*n,l])
a_posteriori_matrix = np.zeros([m,l])

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



#Estimation loop
for iteration in range(500):
    
    #Update of E with random walk
    random_walk += 0.01*np.random.normal(size=([grid_size,grid_size]))
    E = E0+random_walk #Strong noise?
    #E = E0 + np.sin(iteration/(5*np.pi)) + 0.05*np.random.normal(size=([grid_size,grid_size]))
    #b =  0.5*np.random.normal(size=([grid_size,grid_size])) * np.roll(E0, -1, axis=1)
    #E = E0 + b
    E_measured_list.insert(0,E.copy())
    history_current = np.concatenate((E.copy().flatten(),history_current),axis=0)
        
    #Construction of history vectors
    if len(history_current) >= m*n:
        history_list.insert(0,history_current.copy())
        history_current = np.delete(history_current,range(len(history_current)-m,len(history_current)))
        nb_hist +=1
    
    #Check if we need to recompute the filter
    if moving_average > window: 
        already_computed = False
        moving_average = 0
    
    #Construction of the data matrix, a posteriori matrix and computing of the filter  
    if already_computed is False and nb_hist >= l+delta_t:
        for i in range(l):
            data_matrix[:,i] = history_list[i+delta_t]
            a_posteriori_matrix[:,i] = E_measured_list[i].flatten().T
        F = mod.filter_training_full_data(l,m,n,alpha,data_matrix,a_posteriori_matrix)
        #Normalization of the filter
#        a = np.amax(F, axis=1)
#        for i in range(m):
#            F[i,:] = F[i,:] * a[i]
        already_computed = True
    
    #Estimation of E if the filter has been computed
    if already_computed is True:
        E_hat = F.dot(history_list[0]).reshape(grid_size,grid_size)
       # E_hat_list.append(E_hat)
        E_hat_list.insert(0,E_hat.copy())
        moving_average += 1




#Reorder the lists
E_hat_list.reverse() 
E_measured_list.reverse()

##Compute the MSE & Plot the prediction results (reordered lists)
#for i in range(delta_t,len(E_hat_list)):
#    plt.clf()
#    diff = E_hat_list[i] - E_measured_list[i+l+n-1-delta_t]
#    rms_diff = np.abs(E_hat_list[i] - E_measured_list[i+l+n-1-delta_t])/np.abs(E_measured_list[i-delta_t])
#    squared_sum = sum(sum(abs(diff)**2)) #Must account for F??
#    MSE_list.append(squared_sum)
#    
#    plt.subplot(121)
#    plt.imshow(rms_diff, vmin=0, vmax=0.1, cmap='inferno')
#    plt.title('Estimation Error')
#    plt.colorbar()
#    plt.subplot(122)
#    plt.imshow(abs(E_hat_list[i]), cmap='inferno')
#    plt.title('Estimated Field')
#    plt.colorbar()
#    plt.suptitle('Prediction Results')
#    plt.pause(0.1)



#Plot the Estimation for the first pixel
plt.clf()
plt.plot(np.arange(l+n-1-delta_t,len(E_hat_list)+l+n-1-delta_t), [val[0][0] for val in E_hat_list], color='tab:orange', label = 'Estimated Electric Field', linewidth=1)
#plt.plot(np.arange(len(E_hat_list)), 1 ,'r+-', label = 'True position', linewidth=.5)
plt.plot(np.arange(len(E_measured_list)), [val[0][0] for val in E_measured_list], color='tab:blue', label = 'Measured Electric Field', linewidth=1)
plt.ylabel('Value of the first sensor [Ø]')
plt.xlabel('Iteration index k')
plt.title('Predicted value for the first sensor, test data')
plt.legend(bbox_to_anchor = (1, 1), loc = 'upper right', prop = {'size': 10})
plt.show()

#Compute the MSE
for i in range(delta_t,len(E_hat_list)):
    rms_diff = np.abs(E_hat_list[i] - E_measured_list[i+l+n-1-delta_t])#/np.abs(E_measured_list[i+l+n-1-delta_t]
    squared_sum = sum(sum(abs(rms_diff)**2)) 
    MSE_list.append(squared_sum)


#Plot the MSE as a function of iterations
plt.clf()
plt.plot(np.arange(l+n-1-delta_t,len(MSE_list)+l+n-1-delta_t), [np.log(val) for val in MSE_list], color='tab:orange', label = 'MSE', linewidth=1)
#plt.plot(np.arange(len(E_hat_list)), 1 ,'r+-', label = 'True position', linewidth=.5)
plt.plot(np.arange(len(E_measured_list)), [np.log(val[0][0]) for val in E_measured_list], color='tab:blue', label = 'Measured Electric Field', linewidth=1)
plt.ylabel('log(MSE)')
plt.xlabel('Iteration index k')
plt.title('MSE versus measured value')
plt.legend(bbox_to_anchor = (1, 1), loc = 'upper right', prop = {'size': 10})
plt.show()














##Compute the MSE & Plot the prediction results
#for i in range(delta_t,len(E_hat_list)):
#    plt.clf()
#    diff = E_hat_list[i] - E_measured_list[i-delta_t]
#    rms_diff = np.abs(E_hat_list[i] - E_measured_list[i-delta_t])/np.abs(E_measured_list[i-delta_t])
#    squared_sum = sum(sum(abs(diff)**2)) #Must account for F??
#    MSE_list.append(squared_sum)
#    
#    plt.subplot(121)
#    plt.imshow(rms_diff, vmin=0, vmax=0.1, cmap='inferno')
#    plt.title('Estimation Error')
#    plt.colorbar()
#    plt.subplot(122)
#    plt.imshow(abs(E_hat_list[i]), cmap='inferno')
#    plt.title('Estimated Field')
#    plt.colorbar()
#    plt.suptitle('Prediction Results')
#    plt.pause(0.1)
    


##Plot the Estimation for the second pixel
#plt.clf()
#plt.plot(np.arange(l+n-1+delta_t,len(E_hat_list)+l+n-1+delta_t), [val[0][1] for val in E_hat_list],'r+-', label = 'Estimated Electric Field', linewidth=.5)
##plt.plot(np.arange(len(E_hat_list)), 1 ,'r+-', label = 'True position', linewidth=.5)
#plt.plot(np.arange(len(E_measured_list)), [val[0][1] for val in E_measured_list], 'g+-',label = 'Measured Electric Field', linewidth=.5)
#plt.ylabel('Value of the second sensor [Ø]')
#plt.xlabel('Temporal frame [dt]')
#plt.title('Predicted value for the second sensor, test data')
#plt.legend(bbox_to_anchor = (1, 1), loc = 'upper right', prop = {'size': 8})
#plt.show()