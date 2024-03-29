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

TODO:
    Add Pairwise estimation on averaged intensity
    Create Function to add the same noise
    
"""

import hcipy
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



# Create grids
pupil_grid = hcipy.make_pupil_grid(128) 
focal_grid = hcipy.make_focal_grid(4, 16) 
prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)

aperture = hcipy.circular_aperture(1)(pupil_grid)
dark_zone = ((hcipy.circular_aperture(2 * owa)(focal_grid) - hcipy.circular_aperture(2 * iwa)(focal_grid)) * (focal_grid.x > offset)).astype(bool)


# Create optical elements
coronagraph = hcipy.PerfectCoronagraph(aperture)

remove_modes = hcipy.make_zernike_basis(5, 1, pupil_grid, 2)

aberration = hcipy.SurfaceAberration(pupil_grid, 0.1, 1, aperture=aperture, remove_modes=remove_modes)

influence_functions = hcipy.make_xinetics_influence_functions(pupil_grid, num_actuators_across, actuator_spacing)
deformable_mirror = hcipy.DeformableMirror(influence_functions)



# Define functions for getting electric field and intensity from optical system
def get_electric_field(actuators=None):
	if actuators is not None:
		deformable_mirror.actuators = actuators

	wf = hcipy.Wavefront(aperture) 
	img = prop(coronagraph(deformable_mirror(aberration(wf))))
	img_nocoro = prop(deformable_mirror(aberration(wf)))
	
	return img.electric_field / np.abs(img_nocoro.electric_field).max()

def get_intensity(actuators=None):
	if actuators is not None:
		deformable_mirror.actuators = actuators
	
	wf = hcipy.Wavefront(aperture)
	img = prop(coronagraph(deformable_mirror(aberration(wf))))
	img_nocoro = prop(deformable_mirror(aberration(wf)))
	
	return img.intensity / img_nocoro.intensity.max()

def get_noisy_electric_field(random_walk, actuators=None):
    if actuators is not None:
        deformable_mirror.actuators = actuators
        
    wf = hcipy.Wavefront(aperture*np.exp(1j * random_walk))
    img = prop(coronagraph(deformable_mirror(aberration(wf))))
    img_nocoro = prop(deformable_mirror(aberration(wf))) 
	
    return img.electric_field / np.abs(img_nocoro.electric_field).max()



# Define function for plotting
def reshape_dark_hole_to_aperture(vector):
    """ Convert intensity 1D vector to a 2D grid on the focal plane, with zero padding as needed """
    estimate = hcipy.Field(np.zeros(focal_grid.size, dtype='complex'), focal_grid)
    estimate[dark_zone] = vector
    return estimate


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
efc_matrix = hcipy.inverse_tikhonov(response_matrix, 1e-3)

current_actuators = np.zeros(len(influence_functions))



# Create probes
#probes = []
#
#for i in range(4):
#    values = np.sinc(owa * pupil_grid.x) * np.sinc(2 * owa * pupil_grid.y) \
#    * np.cos(2 * np.pi * owa / 2 * pupil_grid.x + i * np.pi / 4)
#    probes.append( hcipy.Field(values, pupil_grid) )

#for i in range(4):
#    plt.subplot(2,2,i+1)
#    hcipy.imshow_field(probes[i])
#plt.show()  



# Filter parameters initialization
l = 3 
m = sum(dark_zone) # Estimate E for every pixel in dark zone
n = 2 
delta_t = 1 
alpha = 1e-3 
if alpha < 0:
  raise Exception("Sorry, regularization parameter must be positive of null") 
window = 2
already_computed = False 
moving_average = 0
nb_hist = 0 



# Filter lists initialization (after grids because we need the size of dark_zone/len actuators)
data_matrix = np.zeros([m*n,l])
a_posteriori_matrix = np.zeros([m,l])
history_current = []
history_list = []
E_measured_list = []
E_hat_list = []
MSE_list = []
rms_diff_list=[]
est_actuators = np.zeros(sum(dark_zone))


    
# Initialize drift parameters
created_dark_hole = False
random_walk = hcipy.Field(np.zeros(aperture.size, dtype='complex'), aperture)

E_avg = np.zeros([sum(dark_zone)], dtype='complex')
E_no_correction_avg = np.zeros([sum(dark_zone)], dtype='complex')

observation_time = 2
count = 0

contrast_corrected_list = []
contrast_uncorrected_list = []








def probe_intensity(probe, actuators=None):
    if actuators is not None:
        deformable_mirror.actuators = actuators
    #dtype = 'complex'
    E0 = hcipy.Field(np.ones(aperture.size), aperture) * 0.001*np.random.normal(size=aperture.size)
    a = E0 + aperture * probe
    wf = hcipy.Wavefront(a)
    img = prop(coronagraph(deformable_mirror(aberration(wf))))
    img_nocoro = prop(deformable_mirror(aberration(wf)))
    
    return img.intensity / img_nocoro.intensity.max()



def take_electric_field_pairwise(probes):
    I_deltas = []
    for probe in probes:
        I_pos = probe_intensity(probe)
        I_neg = probe_intensity(- probe)
        I_deltas.append((I_pos - I_neg)[dark_zone])
    
    return I_deltas
    
#    response_matrix = np.concatenate((response_vector.real, response_vector.imag), axis=1)
#    electric_field_vector = np.linalg.pinv(response_matrix).dot(intensity_vector)
#    
#    return electric_field_vector









# Run EFC loop
for i in range(100):
    # Dark hole creation USING NON ACCESSIBLE PERFECT ELECTRIC FIELD (negligeable noise regime)
    while ( np.mean(np.log10(get_intensity(current_actuators)[dark_zone])) > -9 ) and created_dark_hole is False:
        E = get_electric_field(current_actuators)[dark_zone]
        x = np.concatenate((E.real, E.imag))
        
        y = efc_matrix.dot(x)
        current_actuators -= efc_loop_gain * y
        	
        img = get_intensity(current_actuators)
        
        plt.clf()
        hcipy.imshow_field(np.log10(img), vmin=-10, vmax=-5, cmap='inferno')
        plt.title('Dark hole creation')
        plt.colorbar()
        plt.draw()
        plt.pause(0.05)
                
        if np.mean(np.log10(get_intensity(current_actuators)[dark_zone])) < -9 :
            created_dark_hole = True
            fixed_actuators = current_actuators
        
    # Dark hole maintenance with drifting phase on electric field
    if i < observation_time*(count+1):
        random_walk += hcipy.Field(0.0001*np.random.normal(size=aperture.size), aperture)
        E = get_noisy_electric_field(random_walk, current_actuators)[dark_zone]
        E_no_correction  = get_noisy_electric_field(random_walk, fixed_actuators)[dark_zone]
        # Averaging over the time frames
        E_avg += E
        E_no_correction_avg += E_no_correction
    
    else:
        # Pairwise here to get E from I
        x = np.concatenate((E.real, E.imag))
        y = efc_matrix.dot(x)
        current_actuators -= efc_loop_gain * y
        img = get_intensity(current_actuators)
        img_no_correction = np.abs(E_no_correction)**2
        
        estimate = reshape_dark_hole_to_aperture(E_no_correction_avg)
        
        plt.clf()
        plt.subplot(121)
        hcipy.imshow_field(np.log10(img), vmin=-10, vmax=-5, cmap='inferno')
        plt.title('Dark hole maintenance with additive drift \n images averaged on {} frames, iteration {}'.format(observation_time,i))
        plt.colorbar()
        plt.draw()        
        
        plt.subplot(122)
        hcipy.imshow_field(estimate/observation_time, cmap='inferno')
        plt.colorbar()
        plt.title('Dark hole drift, iteration {}'.format(i))
        plt.draw()
        plt.pause(0.1)
        
        contrast_corrected_list.append(np.log10(np.mean(img[dark_zone])))
        contrast_uncorrected_list.append(np.log10(np.mean(estimate)))
        
        E_avg = np.zeros([sum(dark_zone)], dtype='complex')
        E_no_correction_avg = np.zeros([sum(dark_zone)], dtype='complex')
        count +=1
        
        

plt.clf()
plt.plot(contrast_corrected_list, color='tab:blue', label = 'Corrected contrast', linewidth=1)
plt.plot(contrast_uncorrected_list, color='tab:orange', label = 'Uncorrected contrast', linewidth=1)
plt.legend(bbox_to_anchor = (1, 1), loc = 'right', prop = {'size': 7})
plt.show()

    
    
## Drifting dark hole maintenance with Pairwise and extrapolation with EFC
#for i in range(200):
#    E = get_electric_field(current_actuators)[dark_zone]
#    x = np.concatenate((E.real, E.imag))
#    
#    y = efc_matrix.dot(x)
#    current_actuators -= efc_loop_gain * y
#    
#    E = current_actuators
#    
#    #Run EOF loop on E (data_type = 'complex') or on actuators (data_type = 'real') ????
#    E_measured_list.insert(0,E.copy())
#    history_current = np.concatenate((E.copy().flatten(),history_current),axis=0)
#        
#    #Construction of history vectors
#    if len(history_current) == m*n:
#        history_list.insert(0,history_current.copy())
#        history_current = np.delete(history_current,range(len(history_current)-m,len(history_current)))
#        nb_hist +=1
#        
#    #Check if we need to recompute the filter
#    if window is not None: 
#        if moving_average == window: 
#            already_computed = False
#            moving_average = 0    
#    
#    #Construction of the data matrix, a posteriori matrix and computing of the filter  
#    if already_computed is False and nb_hist >= l+delta_t:
#        for i in range(l):
#            data_matrix[:,i] = history_list[i+delta_t]
#            a_posteriori_matrix[:,i] = E_measured_list[i].flatten()
#        F = eof.filter_training(l,m,n,alpha,data_matrix,a_posteriori_matrix,data_type='real') 
#        already_computed = True
#    
#    #Estimation of E if the filter has been computed
#    if already_computed is True:
#        E_hat = F.dot(history_list[0])#.reshape(grid_size,grid_size)
#        E_hat_list.insert(0,E_hat.copy())
#        moving_average += 1
#    
#    
#    img = get_intensity(current_actuators)
#    
#    if already_computed:
#        
#        E = get_electric_field(E_hat)[dark_zone]
#        x = np.concatenate((E.real, E.imag))
#    
#        y = efc_matrix.dot(x)
#        est_actuators -= efc_loop_gain * y
#        est = get_intensity(est_actuators)
#        
#        plt.clf()
#        plt.subplot(121)
#        imshow_field(np.log10(img), vmin=-10, vmax=-5, cmap='inferno')
#        plt.colorbar()
#        plt.title('Without estimation')
#        plt.draw()
#        
#        plt.subplot(122)
#        imshow_field(np.log10(est), vmin=-10, vmax=-5, cmap='inferno')
#        plt.colorbar()
#        plt.title('With estimation')
#        plt.draw()
#        plt.pause(0.1)
#    
#    else:
#        plt.clf()
#        plt.subplot(121)
#        imshow_field(np.log10(img), vmin=-10, vmax=-5, cmap='inferno')
#        plt.colorbar()
#        plt.title('Without estimation')
#        plt.draw()
#        plt.pause(0.1)
