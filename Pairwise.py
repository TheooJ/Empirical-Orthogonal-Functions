import hcipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Input parameters
num_actuators_across = 32
actuator_spacing = 1.05 / num_actuators_across
epsilon = 1e-5

iwa = 3
owa = 12
offset = 1

efc_loop_gain = 1

# Create grids
pupil_grid = hcipy.make_pupil_grid(128) #128 pixels by dimension, diam = 1
focal_grid = hcipy.make_focal_grid(4, 16)
prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid) 

aperture = hcipy.circular_aperture(1)(pupil_grid) #Evaluated on pupil_grid
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
	
	wf1 = hcipy.Wavefront(aperture)
	img = prop(coronagraph(deformable_mirror(aberration(wf1))))
	img_nocoro = prop(deformable_mirror(aberration(wf1))) 
	
	return img.intensity / img_nocoro.intensity.max()


def get_noisy_electric_field(random_walk, actuators=None):
	if actuators is not None:
		deformable_mirror.actuators = actuators
	
	wf = hcipy.Wavefront(aperture*np.exp(random_walk * 1j))
	img = prop(coronagraph(deformable_mirror(aberration(wf))))
	img_nocoro = prop(deformable_mirror(aberration(wf))) 
	
	return img.electric_field / np.abs(img_nocoro.electric_field).max()


# Creating Jacobian matrix
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

# Run EFC loop
current_actuators = np.zeros(len(influence_functions))



# Define function for plotting an electric field on the focal grid
def reshape_dark_hole_to_aperture(vector):
    """ Convert electric field 1D vector to a 2D grid on the focal plane, with zero padding as needed """
    estimate = hcipy.Field(np.zeros(focal_grid.size, dtype='complex'), focal_grid)
    estimate[dark_zone] = vector
    return estimate

# Define function for plotting an intensity on the focal grid
def reshape_dark_hole_to_aperture_intensity(vector):
    """ Convert intensity 1D vector to a 2D grid on the focal plane, with zero padding as needed """
    estimate = hcipy.Field(np.zeros(focal_grid.size), focal_grid)
    estimate[dark_zone] = vector
    return estimate


# Create probes
probes = []

for i in range(4):
    values = np.sinc(owa * pupil_grid.x) * np.sinc(2 * owa * pupil_grid.y) \
    * np.cos(2 * np.pi * owa / 2 * pupil_grid.x + i * np.pi / 4)
    probes.append(hcipy.Field(values, pupil_grid))

for i in range(4):
    plt.subplot(2,2,i+1)
    hcipy.imshow_field(probes[i])
plt.show()  



def probe_intensity(probe, actuators=None):
    if actuators is not None:
        deformable_mirror.actuators = actuators
    
    aperture_1 = hcipy.circular_aperture(1)(pupil_grid) #Evaluated on pupil_grid
    aberration_1 = hcipy.SurfaceAberration(pupil_grid, 0.1, 1, aperture=aperture_1, remove_modes=remove_modes)

    #dtype = 'complex'
    E0 = hcipy.Field(np.ones(aperture_1.size), aperture_1) * 1 * np.ones(aperture_1.size) # * np.random.normal(size=aperture.size)
    a = aperture_1 * E0 + aperture_1 * probe
    wf_1 = hcipy.Wavefront(a)
    img = prop(coronagraph(deformable_mirror(aberration_1(wf_1))))
    img_nocoro = prop(deformable_mirror(aberration_1(wf_1)))
    
    return img.intensity / img_nocoro.intensity.max()

def probe_electric_field(probe, actuators=None):
    if actuators is not None:
        deformable_mirror.actuators = actuators
    #dtype = 'complex'
    a = aperture * probe
    wf = hcipy.Wavefront(a)
    img = prop(coronagraph(deformable_mirror(aberration(wf))))
    img_nocoro = prop(deformable_mirror(aberration(wf)))
    
    return img.electric_field / np.abs(img_nocoro.electric_field).max()



                             ####### H ou 4*H ???? rcond???
                             
                             
                             
E_actual = get_electric_field()[dark_zone]



def make_observation_matrix(probes):
    H = []
    
    # Collect delta electric field for each probe
    for probe in probes:
        E_pos = probe_electric_field(probe)
        E_neg = probe_electric_field(- probe)

        E_delta = (E_pos - E_neg)
        H.append([E_delta.real, E_delta.imag])

    # Accumulate all tensors in a tensor hcipy.Field
    H = 2 * hcipy.Field(H, focal_grid)

    # Invert the tensors for each pixel with regularization
    # to get the observation matrix.
    H_inv = hcipy.field_inverse_tikhonov(H, 1e-3)[..., dark_zone]
    
    return H_inv

def take_electric_field_pairwise(probes, H_inv):
    I_deltas = []
    
    # Collect delta intensity images for each probe
    for probe in probes:
        I_pos = probe_intensity(probe)
        I_neg = probe_intensity(- probe)
        I_deltas.append((I_pos - I_neg)[dark_zone])
        
    I_deltas = hcipy.Field(np.array(I_deltas), focal_grid)
    
        # Mutiply the observation matrix by the intensity differences for each pixel.
    E = hcipy.Field(np.zeros(focal_grid.size, dtype='complex'), focal_grid)
    y = hcipy.field_dot(hcipy.Field(H_inv, focal_grid), I_deltas)

    # Rebuild electric field from the vector hcipy.Field, and put into dark zone pixels.
    E[dark_zone] = y[0, :] + 1j * y[1, :]

    return E, I_pos



#Compute observed and estimated electric fields
H_inv = make_observation_matrix(probes)
E_estimated , I = take_electric_field_pairwise(probes, H_inv)
E_estimated = E_estimated[dark_zone]


E_actual_energy = np.linalg.norm(E_actual) ** 2
E_estimated_energy = np.linalg.norm(E_estimated) ** 2
energy_difference = (np.linalg.norm(E_actual - E_estimated) ** 2) / E_actual_energy

E_diff = np.abs(E_actual - E_estimated) **2 / E_actual_energy



#Print electric fields
plt.clf()
plt.subplot(121)
hcipy.imshow_field(reshape_dark_hole_to_aperture(E_actual), cmap='inferno')
plt.title('Actual electric field')
plt.colorbar()
plt.draw()        

plt.subplot(122)
hcipy.imshow_field(reshape_dark_hole_to_aperture(E_estimated), cmap='inferno')
plt.colorbar()
plt.title('Estimated electric field')
plt.draw()



#Print intensities for verification

#Scott (absolute difference)
plt.figure()
hcipy.imshow_field(reshape_dark_hole_to_aperture_intensity((np.abs(E_actual - E_estimated) ** 2)), norm=colors.LogNorm(vmin=1e-8, vmax=1e-1))
plt.colorbar()
plt.title('actual - pairwise')

#Moi (relative difference)
plt.figure()
hcipy.imshow_field(reshape_dark_hole_to_aperture_intensity(E_diff), norm=colors.LogNorm(vmin=1e-8, vmax=1e-1))
plt.colorbar()
plt.title('(actual - pairwise) / actual')



##random_walk = Field(0.001*np.random.normal(size=aperture.size), aperture) * 1j
#random_walk = Field(np.zeros(aperture.size, dtype='complex'), aperture)
#for i in range(100):
#    if i<25:
#        E = get_noisy_electric_field(random_walk, current_actuators)[dark_zone]
#        x = np.concatenate((E.real, E.imag))
#        
#        y = efc_matrix.dot(x)
#        current_actuators -= efc_loop_gain * y
#        	
#        img = get_intensity(current_actuators)
#        
#        plt.clf()
#        imshow_field(np.log10(img), vmin=-10, vmax=-5, cmap='inferno')
#        plt.colorbar()
#        plt.title('Dark hole creation, iteration {}'.format(i))
#        plt.draw()
#        plt.pause(0.1)
#        
#    else: 
#        random_walk += Field(0.001*np.random.normal(size=aperture.size), aperture)
#        E = get_noisy_electric_field(random_walk, current_actuators)[dark_zone]
#        x = np.concatenate((E.real, E.imag))
#        
#        y = efc_matrix.dot(x)
#        current_actuators -= efc_loop_gain * y
#        	
#        img = get_intensity(current_actuators)
#        
#        plt.clf()
#        imshow_field(np.log10(img), vmin=-10, vmax=-5, cmap='inferno')
#        plt.colorbar()
#        plt.title('Dark hole creation, iteration {}'.format(i))
#        plt.draw()
#        plt.pause(0.1)