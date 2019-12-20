'''
Numerical solutions to the 2D N-body problem
'''

import json
import time
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

import caching
# Import main physical functions from dedicated module
from physics import derivatives, calc_total_energy, calc_total_ang_momentum
from tools import pprint

# Constants
# FILE_NAME = 'HD3651'
DATA_DIR = 'data'
INDEX_FILE = 'index'

D = 3  # Number of spatial dimensions


# Functions


def load_bodies_from_json(file_name='bodies'):
    '''
    Read initial conditions of bodies from json file.
    * @param file_name name of file on system to parse

    Returns tuple of masses and vectorised form of initial conditions
    '''
    with open(f'bodies/{file_name}.json', 'r') as bodies_handler:
        # Catch any errors in the JSON files and alert
        try:
            dump = json.loads(bodies_handler.read())
        except json.decoder.JSONDecodeError:
            raise SyntaxError(f'The file {file_name}.json has an error')

    initial_values = dump['initial_values']

    masses = np.array([body['m'] for body in initial_values], ndmin=2)
    names = np.array([body.get('name') for body in initial_values])

    labels = ['x', 'y', 'z', 'v0x', 'v0y', 'v0z']

    # Vectorise initial values
    initial_state = [[body[l] for body in initial_values] for l in labels]

    return (dump['G'], masses, np.concatenate(initial_state), dump['tf'],
            dump['tmax'], names)


def format_masses_as_names(masses):
    '''
    Format masses values for legend labels.
    * @param masses iterable of mass values

    Returns Iterable of str
    '''
    return [f'$m_{i} = {m}$' for i, m in enumerate(masses)]


def animate_plot(ax, dims, masses, names, x, y, z, times, tf, speed):
    # Animate in real time, 1s == 1 time unit
    toc = tic = time.time()
    while toc - tic < tf / speed:
        ax.clear()
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))

        if dims == 3:
            z_min, z_max = np.min(z), np.max(z)
            if not np.isclose(z_min, z_max):
                ax.set_zlim(z_min, z_max)
        else:
            ax.axis('equal')

        # Closest index in data to current time value
        idx = int(len(times) * (toc - tic) / (tf / speed))
        for j in range(masses.size):
            # Optionally add z values if 3D plot is chosen
            if dims == 3:
                zs = z[j][:idx],
                zt = [[z[j][idx]]]
            else:
                # Empty iterable results in nothing passed on broadcast
                zs = zt = ()
            l, = ax.plot(x[j][:idx], y[j][:idx], *zs)

            # Use same colour from line in marker plot
            ax.plot([coords[j][idx]], [coords[masses.size + j][idx]],
                    *zt, marker='o', c=l.get_c(), label=names[j])

        ax.legend()

        plt.pause(1e-5)
        toc = time.time()


def draw_bodies(masses, times, state, *, names=None, dims=2, animate=False,
                speed=1, tf=None, ax=None, fig=None):
    '''
    Plot trajectories of the bodies.
    * @param masses numpy array of the masses
    * @param times numpy array of time values
    * @param state numpy array of vectorised coordinates
        - rows are the coordinate index
        - columns are the time index
    * @param animate whether to animate the trajectories
    * @param tf duration of animation
    * @param ax axis to plot onto
    * @param fig figure window to redraw
    '''

    # Extract the space coordinates from the state
    x, y, z, *_ = np.split(state, 2*D)

    # Allow passing of own axis to adjust plot
    if ax is None or fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d' if dims == 3 else None)
    ax.margins(x=0, y=0)

    if animate and tf is None:
        raise ValueError('Provide duration tf to animate')

    if names is None:
        names = format_masses_as_names(masses[0])
    elif not all(names):
        names = format_masses_as_names(masses[0])

    print('Drawing...')
    if animate:
        animate_plot(ax, dims, masses, names, x, y, z, times, tf, speed)
    else:
        for j, m in enumerate(masses[0]):
            ax.plot(coords[j], coords[masses.size + j], label=names[j])
        ax.legend()
        ax.set_title('Trajectories')
    print('Finished drawing')


def draw_stats(gravity, masses, times, state, *, rel_L=True, rel_E=True,
               axs=None, fig=None):
    '''
    Calculate & plot the total energy and angular momenta for all times.
    * @param gravity Universal gravitational constant is chosen units
    * @param masses numpy array of the masses
    * @param times numpy array of time values
    * @param coords numpy array of vectorised coordinates
        - rows are the coordinate index
        - columns are the time index
    * @param rel_L whether to plot angular momentum relative to the first value
    * @param rel_E whether to plot total energy relative to the first value
    * @param ax axes to plot onto. First momenta, second energy
    * @param fig figure window to redraw
    '''

    if axs is None or fig is None:
        fig, (momenta_ax, energy_ax) = plt.subplots(ncols=2)
    else:
        momenta_ax, energy_ax = axs
    momenta_ax.margins(x=0)
    energy_ax.margins(x=0)

    E = calc_total_energy(gravity, D, masses, state)
    L = calc_total_ang_momentum(D, masses, state)

    if rel_L:
        momenta_ax.plot(times, L / L[0] - 1, label='$L / L_0 - 1$')
    else:
        momenta_ax.plot(times, L, label='$L$')

    if rel_E:
        energy_ax.plot(times, E / E[0] - 1, label='$E / E_0 - 1$')
    else:
        energy_ax.plot(times, E, label='$E$')

    momenta_ax.legend()
    energy_ax.legend()
    momenta_ax.set_title('Angular Momentum')
    energy_ax.set_title('Total Energy')


def solve_for(file_name, calc=False):
    '''
    Solves the N-body problem for initial values in json file.
    * @param file_name path without extension to extract initial values
    * @param calc (Re)Calculate even if there is a cashed version
    '''
    # Vectorise the initial values and extract parameters
    (gravity,
     masses,
     initial_values,
     tf,
     tmax,
     names) = load_bodies_from_json(file_name)

    hash_ = caching.get_hash(gravity, masses, initial_values, tf, tmax)

    # Load the trajectory from file if it has already been solved
    # Otherwise solve with initial conditions and write to file
    if hash_ in caching.parse_index(INDEX_FILE) and not calc:
        print('Loading trajectories from file...')
        times, coords = caching.load_data(DATA_DIR, hash_)
    else:
        print('Calculating trajectories...')

        d = partial(derivatives, D, gravity, masses)

        tic = time.time()
        sol = solve_ivp(d, [0, tf], initial_values, max_step=tmax)
        toc = time.time()
        print(f'Solved in {toc - tic:g}')

        times = sol.t
        coords = sol.y

        if not calc:
            caching.create_data_dir(DATA_DIR)
            caching.write_data(hash_, DATA_DIR, times, coords)
            caching.update_index(INDEX_FILE, hash_)

    return gravity, masses, times, coords, tf, names


if __name__ == '__main__':
    file_name = input('Initial values file name: ')
    # file_name = FILE_NAME
    gravity, masses, times, coords, tf, names = solve_for(file_name)

    draw_bodies(masses, times, coords, tf=tf, names=names)
    draw_stats(gravity, masses, times, coords)

    plt.show()
