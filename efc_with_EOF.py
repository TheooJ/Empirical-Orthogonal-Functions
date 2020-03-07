"""Implement EOF on a coronagraph simulation using HciPy.

Construct an optical system with one DM, estimate the speckles electric field 
with pairwise and EOF. Improve contrast with EFC and compare results.


Optical parameters:
:param num_actuators_across: Number of actuators in one dimension

Filter parameters: 
:param l: Number of history vectors in data matrix
:param m: Number of measurements in history vectors 
:param n: Number of time samples in history vectors 
:param delta_t: Number of steps in the future for which the estimation will happen
:param alpha: Tikhonov regularization of the filter
:param window: Number of samples before recomputing the filter \
               Set to None to only compute the filter once
:data_type: Type of the training data. Either 'real' or 'complex'.
"""

from hcipy import *
import numpy as np
import matplotlib.pyplot as plt
import empirical_orthogonal_functions as eof


# Input parameters
num_actuators_across = 32
actuator_spacing = 1.05 / num_actuators_across
epsilon = 1e-5

iwa = 3
owa = 12
offset = 1

efc_loop_gain = 1

#Filter parameters initialization
l = 3 
#m = 4   #Chosen later
n = 2 
delta_t = 1 
alpha = 1e-3 
if alpha < 0:
  raise Exception("Sorry, regularization parameter must be positive of null") 
window = 2
already_computed = False 
moving_average = 0
nb_hist = 0 



# Create grids
pupil_grid = make_pupil_grid(128) 
focal_grid = make_focal_grid(4, 16) 
prop = FraunhoferPropagator(pupil_grid, focal_grid)

aperture = circular_aperture(1)(pupil_grid)
dark_zone = ((circular_aperture(2 * owa)(focal_grid) - circular_aperture(2 * iwa)(focal_grid)) * (focal_grid.x > offset)).astype(bool)


# Create optical elements
coronagraph = PerfectCoronagraph(aperture)

remove_modes = make_zernike_basis(5, 1, pupil_grid, 2)

aberration = SurfaceAberration(pupil_grid, 0.1, 1, aperture=aperture, remove_modes=remove_modes)

influence_functions = make_xinetics_influence_functions(pupil_grid, num_actuators_across, actuator_spacing)
deformable_mirror = DeformableMirror(influence_functions)



# Define functions for getting electric field and intensity from optical system
def get_electric_field(actuators=None):
	if actuators is not None:
		deformable_mirror.actuators = actuators

	wf = Wavefront(aperture) ###drift
	img = prop(coronagraph(deformable_mirror(aberration(wf))))
	img_nocoro = prop(deformable_mirror(aberration(wf)))
	
	return img.electric_field / np.abs(img_nocoro.electric_field).max()

def get_intensity(actuators=None):
	if actuators is not None:
		deformable_mirror.actuators = actuators
	
	wf = Wavefront(aperture)
	img = prop(coronagraph(deformable_mirror(aberration(wf))))
	img_nocoro = prop(deformable_mirror(aberration(wf)))
	
	return img.intensity / img_nocoro.intensity.max()



# Create Jacobian matrix
responses = []
amps = np.linspace(-epsilon, epsilon, 2)

for i, mode in enumerate(np.eye(len(influence_functions))):
	print(i, '/', len(influence_functions))
	response = 0
	
	for amp in amps:
		response += amp * get_electric_field(mode * amp)
		
	response /= np.var(amps)
	response = response[dark_zone]

	responses.append(np.concatenate((response.real, response.imag)))

    
    
# Calculate EFC matrix
response_matrix = np.array(responses).T
efc_matrix = inverse_tikhonov(response_matrix, 1e-3)



# Run EFC loop
current_actuators = np.zeros(len(influence_functions))
est_actuators = np.zeros(len(influence_functions))

#m = sum(dark_zone) #Estimate E for every pixel in dark zone
m = len(current_actuators)


#Filter lists initialization (after grids because we need the size of dark_zone/len actuators)
data_matrix = np.zeros([m*n,l])
a_posteriori_matrix = np.zeros([m,l])
history_current = []
history_list = []
E_measured_list = []
E_hat_list = []
MSE_list = []
rms_diff_list=[]


for i in range(200):
    E = get_electric_field(current_actuators)[dark_zone]
    x = np.concatenate((E.real, E.imag))
    
    y = efc_matrix.dot(x)
    current_actuators -= efc_loop_gain * y
    
    E = current_actuators
    
    #Run EOF loop on E (data_type = 'complex') or on actuators (data_type = 'real') ????
    E_measured_list.insert(0,E.copy())
    history_current = np.concatenate((E.copy().flatten(),history_current),axis=0)
        
    #Construction of history vectors
    if len(history_current) == m*n:
        history_list.insert(0,history_current.copy())
        history_current = np.delete(history_current,range(len(history_current)-m,len(history_current)))
        nb_hist +=1
        
    #Check if we need to recompute the filter
    if window is not None: 
        if moving_average == window: 
            already_computed = False
            moving_average = 0    
    
    #Construction of the data matrix, a posteriori matrix and computing of the filter  
    if already_computed is False and nb_hist >= l+delta_t:
        for i in range(l):
            data_matrix[:,i] = history_list[i+delta_t]
            a_posteriori_matrix[:,i] = E_measured_list[i].flatten()
        F = eof.filter_training(l,m,n,alpha,data_matrix,a_posteriori_matrix,data_type='real') 
        already_computed = True
    
    #Estimation of E if the filter has been computed
    if already_computed is True:
        E_hat = F.dot(history_list[0])#.reshape(grid_size,grid_size)
        E_hat_list.insert(0,E_hat.copy())
        moving_average += 1
    
    
    img = get_intensity(current_actuators)
    
    if already_computed:
        
        E = get_electric_field(E_hat)[dark_zone]
        x = np.concatenate((E.real, E.imag))
    
        y = efc_matrix.dot(x)
        est_actuators -= efc_loop_gain * y
        est = get_intensity(est_actuators)
        
        plt.clf()
        plt.subplot(121)
        imshow_field(np.log10(img), vmin=-10, vmax=-5, cmap='inferno')
        plt.colorbar()
        plt.title('Without estimation')
        plt.draw()
        
        plt.subplot(122)
        imshow_field(np.log10(est), vmin=-10, vmax=-5, cmap='inferno')
        plt.colorbar()
        plt.title('With estimation')
        plt.draw()
        plt.pause(0.1)
    
    else:
        plt.clf()
        plt.subplot(121)
        imshow_field(np.log10(img), vmin=-10, vmax=-5, cmap='inferno')
        plt.colorbar()
        plt.title('Without estimation')
        plt.draw()
        plt.pause(0.1)
