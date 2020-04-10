"""
Implement EOF on HciPy.

Code to estimate the electric field on focal plane using optical elements.
After the dark zone is created with efc, add a random walk on the incident phase.
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
pupil_grid = hcipy.make_pupil_grid(128)  # 128 pixels by dimension, diam = 1
focal_grid = hcipy.make_focal_grid(4, 16)
prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)

aperture = hcipy.circular_aperture(1)(pupil_grid)  # Evaluated on pupil_grid
dark_zone = ((hcipy.circular_aperture(2 * owa)(focal_grid) - hcipy.circular_aperture(2 * iwa)(focal_grid)) * (
            focal_grid.x > offset)).astype(bool)

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


# Define functions to simulate random walk
def get_noisy_electric_field(random_walk, actuators=None):
    if actuators is not None:
        deformable_mirror.actuators = actuators

    wf = hcipy.Wavefront(aperture * np.exp(random_walk * 1j))
    img = prop(coronagraph(deformable_mirror(aberration(wf))))
    img_nocoro = prop(deformable_mirror(aberration(wf)))

    return img.electric_field / np.abs(img_nocoro.electric_field).max()


def get_noisy_intensity(random_walk, actuators=None):
    if actuators is not None:
        deformable_mirror.actuators = actuators

    wf = hcipy.Wavefront(aperture * np.exp(random_walk * 1j))
    img = prop(coronagraph(deformable_mirror(aberration(wf))))
    img_nocoro = prop(deformable_mirror(aberration(wf)))

    return img.intensity / img_nocoro.intensity.max()


# Define functions for plotting a dark zone on the focal grid
def reshape_dark_hole_to_aperture(vector):
    """ Convert electric field 1D vector to a 2D grid on the focal plane, with zero padding as needed """
    estimate = hcipy.Field(np.zeros(focal_grid.size, dtype='complex'), focal_grid)
    estimate[dark_zone] = vector
    return estimate


def reshape_dark_hole_to_aperture_intensity(vector):
    """ Convert intensity 1D vector to a 2D grid on the focal plane, with zero padding as needed """
    estimate = hcipy.Field(np.zeros(focal_grid.size), focal_grid)
    estimate[dark_zone] = vector
    return estimate


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

# Filter parameters initialization
l = 2
m = sum(dark_zone)  # Estimate E for every pixel in dark zone
n = 2
delta_t = 1
alpha = 0
if alpha < 0:
    raise Exception("Sorry, regularization parameter must be positive of null")
window = 4
already_computed = False
moving_average = 0
nb_hist = 0

# Filter lists initialization (after grids because we need the size of dark_zone/len actuators)
data_matrix = np.zeros([m * n, l], dtype='complex')
a_posteriori_matrix = np.zeros([m, l], dtype='complex')
history_current = []
history_list = []
E_measured_list = []
E_hat_list = []

# Initialize drift parameters
created_dark_hole = False
random_walk = hcipy.Field(np.zeros(aperture.size), aperture)  # , dtype='complex'
contrast_list = []

# Run dark zone creation with efc then eof estimation
for iteration in range(50):
    # Dark hole creation USING NON ACCESSIBLE PERFECT ELECTRIC FIELD (negligible noise regime)
    while (np.mean(np.log10(get_intensity(current_actuators)[dark_zone])) > -9) and created_dark_hole is False:
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

        if np.mean(np.log10(get_intensity(current_actuators)[dark_zone])) < -9:
            created_dark_hole = True

    # Estimation when dark zone has been created
    random_walk += hcipy.Field(0 * np.random.normal(size=aperture.size), aperture)
    E = get_noisy_electric_field(random_walk, current_actuators)[dark_zone]

    E_measured_list.insert(0, E.copy())
    history_current = np.concatenate((E.copy().flatten(), history_current), axis=0)

    # Construction of history vectors
    if len(history_current) == m * n:
        history_list.insert(0, history_current.copy())
        history_current = np.delete(history_current, range(len(history_current) - m, len(history_current)))
        nb_hist += 1

    # Check if we need to recompute the filter
    if window is not None:
        if moving_average == window:
            already_computed = False
            moving_average = 0

    # Construction of the data matrix, a posteriori matrix and computing of the filter
    if already_computed is False and nb_hist >= l + delta_t:
        for i in range(l):
            data_matrix[:, i] = history_list[i + delta_t]
            a_posteriori_matrix[:, i] = E_measured_list[i].flatten()
        F = eof.filter_training(l, m, n, alpha, data_matrix, a_posteriori_matrix, data_type='complex')
        already_computed = True

    # Estimation of E if the filter has been computed
    if already_computed:
        E_hat = F.dot(history_list[0])
        E_hat_list.insert(0, E_hat.copy())
        I_hat = np.abs(E_hat) ** 2
        estimate = reshape_dark_hole_to_aperture_intensity(I_hat)

        I_actual_energy = np.linalg.norm(img[dark_zone]) ** 2
        energy_difference = (np.linalg.norm(img[dark_zone] - estimate[dark_zone]) ** 2) / I_actual_energy

        print('Energy difference is {}'.format(energy_difference))

        plt.clf()
        hcipy.imshow_field(np.log10(estimate), vmin=-10, vmax=-5, cmap='inferno')
        plt.colorbar()
        plt.title('Dark hole electric field estimate, iteration {}'.format(iteration))
        plt.draw()
        plt.pause(0.1)

        moving_average += 1
