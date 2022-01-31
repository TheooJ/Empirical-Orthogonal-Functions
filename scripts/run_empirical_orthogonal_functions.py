"""Time series analysis using empirical orthogonal functions.

Implement the Empirical Orthogonal Functions from Guyon et al. 2017,
compute the filter F from a matrix of concatenated history data that can be 
regularly updated.

:param grid_size: Size of the image grid to construct
:param l: Number of history vectors in data matrix
:param m: Number of measurements in history vectors 
:param n: Number of time samples in history vectors 
:param delta_t: Number of steps in the future for which the estimation will happen
:param alpha: Tikhonov regularization of the filter
:param window: Number of samples before recomputing the filter \
               Set to None to only compute the filter once
:data_type: Type of the training data. Either 'real' or 'complex'.
"""

import numpy as np
import matplotlib.pyplot as plt
import empirical_orthogonal_functions as eof

# Initialize the parameters
grid_size = 20

l = 10
m = grid_size ** 2
n = 4
delta_t = 1

alpha = 0
if alpha < 0:
    raise Exception("Sorry, regularization parameter must be positive of null")

window = None
already_computed = False
moving_average = 0
nb_hist = 0

data_type = 'real'

# Initialize the matrices
## Uniform
E0 = np.ones([grid_size, grid_size])  # We don't have access to this in reality
data_matrix = np.zeros([m * n, l])
a_posteriori_matrix = np.zeros([m, l])
random_walk = np.zeros([grid_size, grid_size])

## Speckles
center_1 = [5, 5]
center_2 = [15, 15]
sigma_1 = 0.9
sigma_2 = 2.7

planet = np.zeros([grid_size, grid_size])
speckles = np.zeros([grid_size, grid_size])

for row in range(planet.shape[0]):
    for col in range(planet.shape[1]):
        planet[row][col] = np.exp(- np.sqrt((row - center_1[0]) ** 2 + (col - center_1[1]) ** 2) / sigma_1 ** 2)
        speckles[row][col] = np.exp(- np.sqrt((row - center_2[0]) ** 2 + (col - center_2[1]) ** 2) / sigma_2 ** 2)

camera = planet + speckles

# Initialize the lists
history_current = []
history_list = []
E_measured_list = []
E_hat_list = []
MSE_list = []
rms_diff_list = []

# Estimation loop
for iteration in range(200):

    # Update E
    random_walk += 0.05 * np.random.normal(size=([grid_size, grid_size]))
    # E = camera + random_walk + 0.1*np.random.poisson(abs(E0), size=([grid_size,grid_size]))#Drift
    E = camera + random_walk

    # E = E0 + 0.05*np.random.normal(size=([grid_size,grid_size])) #Normal gaussian noise

    # E = E0 + 0.03*np.random.poisson(abs(E0), size=([grid_size,grid_size]))

    # E = E0 + np.sin(iteration/(5*np.pi)) + 0.05*np.random.normal(size=([grid_size,grid_size]))
    # b =  0.03*np.random.normal(size=([grid_size,grid_size])) * np.roll(E0, -1, axis=1)
    # E = E0 + b

    E_measured_list.insert(0, E.copy())
    history_current = np.concatenate((E.copy().flatten(), history_current), axis=0)

    # Construct history vectors
    if len(history_current) == m * n:  # (>=)
        history_list.insert(0, history_current.copy())
        history_current = np.delete(history_current, range(len(history_current) - m, len(history_current)))
        nb_hist += 1

    # Check if we need to recompute the filter
    if window is not None:
        if moving_average == window:
            already_computed = False
            moving_average = 0

    # Construct the data matrix, a posteriori matrix and compute the filter  
    if already_computed is False and nb_hist >= l + delta_t:
        for i in range(l):
            data_matrix[:, i] = history_list[i + delta_t]
            a_posteriori_matrix[:, i] = E_measured_list[i].flatten()
        F = eof.filter_training(l, m, n, alpha, data_matrix, a_posteriori_matrix, data_type)
        already_computed = True

    # Estimate E if the filter has been computed
    if already_computed is True:
        E_hat = F.dot(history_list[0]).reshape(grid_size, grid_size)
        E_hat_list.insert(0, E_hat.copy())
        moving_average += 1

# Reorder the lists
E_hat_list.reverse()
E_measured_list.reverse()

####Compute the MSE & Plot the prediction results (reordered lists)

for i in range(delta_t, len(E_hat_list)):
    plt.clf()
    diff = E_hat_list[i] - E_measured_list[i + l + n - 2 + delta_t]
    rms_diff = np.abs(diff) / np.abs(E_measured_list[i + l + n - 2 + delta_t])
    squared_sum = sum(sum(abs(diff) ** 2))  # Must account for F??
    MSE_list.append(squared_sum)

    plt.subplot(131)
    plt.imshow(abs(E_measured_list[i + l + n - 2 + delta_t]), cmap='inferno')
    plt.title('Measured Field')
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(abs(E_hat_list[i]), cmap='inferno')
    plt.title('Estimated Field')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(rms_diff, vmin=0, vmax=0.1, cmap='inferno')
    plt.title('Estimation Error')
    plt.colorbar()

    plt.suptitle('Prediction Results, iteration {}'.format(i))
    plt.pause(0.1)

# Plot the Estimation for the first pixel
plt.clf()
plt.plot(np.arange(l + n - 2 + delta_t, len(E_hat_list) + l + n - 2 + delta_t), [val[0][0] for val in E_hat_list],
         color='tab:orange', label='Estimated Electric Field', linewidth=1)
plt.plot(np.arange(len(E_measured_list)), [val[0][0] for val in E_measured_list], color='tab:blue',
         label='Measured Electric Field', linewidth=1)
plt.ylabel('Value of the first sensor [Ø]')
plt.xlabel('Iteration index k')
plt.title('Predicted versus measured value for the first sensor')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', prop={'size': 10})
plt.show()

# Compute the MSE
if not (MSE_list):
    for i in range(delta_t, len(E_hat_list)):
        rms_diff = np.abs(
            E_hat_list[i] - E_measured_list[i + l + n - 2 + delta_t])  # /np.abs(E_measured_list[i+l+n-2+delta_t])
        rms_diff_list.append(rms_diff)
        squared_sum = sum(sum(abs(rms_diff) ** 2))
        MSE_list.append(squared_sum)

# Plot the MSE as a function of iterations
plt.clf()
plt.plot(np.arange(l + n - 2 + delta_t, len(MSE_list) + l + n - 2 + delta_t), [np.log(val) for val in MSE_list],
         color='tab:orange', label='log(MSE)', linewidth=1)
plt.plot(np.arange(len(E_measured_list)), [np.log(val[0][0]) for val in E_measured_list], color='tab:blue',
         label='log(Measured Electric Field)', linewidth=1)
plt.ylabel('log(MSE)')
plt.xlabel('Iteration index k')
plt.title('MSE versus measured value')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', prop={'size': 10})
plt.show()

#
#####Plot rmd_diff pour montrer recalcul du filtre
# for i in range(28,34):
#    plt.clf()
#    plt.imshow(rms_diff_list[i], vmin=0, vmax=0.1, cmap='inferno')
#    plt.title('Reconstruction MSE')
#    plt.colorbar()
#    plt.pause(0.1)


# Compute the autocorrelation
autocorrelation_list = []
for i in range(len(history_list)):
    a = sum([x * np.conj(y) for x, y in zip(history_list[0], history_list[i])])
    autocorrelation_list.append(a.copy())

    # TO DO: Divide by nb points

# Plot the autocorrelation
plt.clf()
plt.plot(np.arange(len(autocorrelation_list)), [val for val in autocorrelation_list], color='tab:blue', linewidth=1)
plt.ylabel('Autocorrelation [wf^2]')
plt.xlabel('Time shift')
plt.title('Autocorrelation of the history vector')
plt.show()

print('Autocorrelation slope is', (autocorrelation_list[-1] - autocorrelation_list[0]) / len(autocorrelation_list))
