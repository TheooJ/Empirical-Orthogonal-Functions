from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

# Input parameters
num_actuators_across = 32
actuator_spacing = 1.05 / num_actuators_across
epsilon = 1e-5

iwa = 3
owa = 12
offset = 1
offset_max = 4

efc_loop_gain = 1

# Create grids
pupil_grid = make_pupil_grid(128)  # 128 pixels by dimension, diam = 1
focal_grid = make_focal_grid(4, 16)  # 4,16
prop = FraunhoferPropagator(pupil_grid,
                            focal_grid)  # From pupil to focal grid, assumes perfect fit on front and back focal planes)

aperture = circular_aperture(1)(pupil_grid)  # Evaluated on pupil_grid
# mask = make_uniform_grid([128,128], [1,1],  center=0)
dark_zone = ((circular_aperture(2 * owa)(focal_grid) - circular_aperture(2 * iwa)(focal_grid)) * (focal_grid.x > offset)).astype(bool)
#dark_zone = ((circular_aperture(2 * owa)(focal_grid) - circular_aperture(2 * iwa)(focal_grid)) * (focal_grid.x > offset) * (focal_grid.y > offset) * (focal_grid.x < offset_max) * (focal_grid.y < offset_max)).astype(bool)

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

    wf = Wavefront(aperture)
    img = prop(coronagraph(deformable_mirror(aberration(wf))))
    img_nocoro = prop(deformable_mirror(aberration(wf)))  # For normalization

    return img.electric_field / np.abs(img_nocoro.electric_field).max()


def get_intensity(actuators=None):
    if actuators is not None:
        deformable_mirror.actuators = actuators

    wf = Wavefront(aperture)
    img = prop(coronagraph(deformable_mirror(aberration(wf))))
    img_nocoro = prop(deformable_mirror(aberration(wf)))  # For normalization

    return img.intensity / img_nocoro.intensity.max()


# Creating Jacobian matrix
responses = []
amps = np.linspace(-epsilon, epsilon, 2)

for i, mode in enumerate(np.eye(len(influence_functions))):
    print(i, '/', len(influence_functions))
    response = 0

    for amp in amps:
        response += amp * get_electric_field(mode * amp)  # Fancy derivative

    response /= np.var(amps)
    response = response[dark_zone]

    responses.append(np.concatenate((response.real, response.imag)))

# Calculate EFC matrix
response_matrix = np.array(responses).T
efc_matrix = inverse_tikhonov(response_matrix, 1e-3)

# Run EFC loop
current_actuators = np.zeros(len(influence_functions))

for i in range(30):
    # while ( sum(np.log10(get_intensity(current_actuators)[dark_zone]))/sum(dark_zone) > -10 ):
    E = get_electric_field(current_actuators)[dark_zone]
    x = np.concatenate((E.real, E.imag))

    y = efc_matrix.dot(x)
    current_actuators -= efc_loop_gain * y

    img = get_intensity(current_actuators)

    plt.clf()
    imshow_field(np.log10(img), vmin=-10, vmax=-5, cmap='inferno')
    plt.colorbar()
    plt.draw()
    plt.pause(0.1)
