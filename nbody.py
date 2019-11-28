'''
Numerical solutions to the 2D N-body problem
'''

import json
import time
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from caching import *
from tools import pprint

# Constants
FILE_NAME = 'figureeight'

DATA_DIR = 'data'
INDEX_FILE = 'index'

GRAVITY = 1

d = 3
m = 2*d  # Number of coordinates (2 position 2 speed)
FRAMERATE = 60  # Number of data points to be saved per unit time


# Functions


def get_vars_from_state(state):
    '''
    Slice state into coords and speeds.
    Coords and speed are split into x, y, z components within the lists.
    * @param state List of corrdinates in vector form. Length must be
        integer multiples of m.
    '''
    n = len(state) // m

    coords = np.split(np.array(state[:len(state) // 2]), n)
    speeds = np.split(np.array(state[len(state) // 2:]), n)

    return coords, speeds


def derivatives(masses, time, state):
    '''
    Calculate the values of the speed and acceleration of each body.
    * @param masses
    * @param time current time value
    * @param state vectorised list of variables

    Returns Derivatives in vectorised form
    '''
    (x, y, z), (vx, vy, vz) = get_vars_from_state(state)

    x_separation = x - x.T
    y_separation = y - y.T
    z_separation = z - z.T

    cubed_separation = (x_separation**2 + y_separation **
                        2 + z_separation**2)**1.5
    np.fill_diagonal(cubed_separation, np.nan)

    x_acceleration = -GRAVITY * masses.T * x_separation / cubed_separation
    y_acceleration = -GRAVITY * masses.T * y_separation / cubed_separation
    z_acceleration = -GRAVITY * masses.T * z_separation / cubed_separation

    np.fill_diagonal(x_acceleration, 0)
    np.fill_diagonal(y_acceleration, 0)
    np.fill_diagonal(z_acceleration, 0)

    return np.concatenate((vx, vy, vz, np.sum(x_acceleration, axis=0),
                           np.sum(y_acceleration, axis=0),
                           np.sum(z_acceleration, axis=0)))


def load_bodies_from_json(file_name='bodies'):
    '''
    Read initial conditions of bodies from json file.
    * @param file_name name of file on system to parse

    Returns tuple of masses and vectorised form of initial conditions
    '''
    with open(f'{file_name}.json', 'r') as bodies_handler:
        dump = json.loads(bodies_handler.read())

        initial_values = dump['initial_values']

        masses = np.array([body['m'] for body in initial_values],
                          ndmin=2)

        x = [body['x'] for body in initial_values]
        y = [body['y'] for body in initial_values]
        z = [body['z'] for body in initial_values]
        v0x = [body['v0x'] for body in initial_values]
        v0y = [body['v0y'] for body in initial_values]
        v0z = [body['v0z'] for body in initial_values]

        return (masses, np.concatenate((x, y, z, v0x, v0y, v0z)),
                dump['tf'], dump['tmax'])


def draw_bodies(masses, times, coords, *, tf=None, animate=False, ax=None, fig=None):
    '''
    Plot trajectories of the bodies.
    * @param m asses numpy array of time values
    * @param coords numpy array of vectorised coordinates
        - rows are the coordinate index
        - columns are the time index
    * @param animate whether to animate the trajectories
    '''
    if ax is None or fig is None:
        fig, ax = plt.subplots()
    ax.margins(x=0, y=0)

    n = np.size(masses)

    if animate and tf is None:
        raise ValueError('Provide duration tf to animate')

    if animate:
        print('Drawing...')

        toc = tic = time.time()
        while toc - tic < tf:
            ax.clear()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)

            idx = int(len(times) * (toc - tic) / tf)
            for j in range(n):
                ax.plot(coords[j][idx], coords[n + j][idx], 'o')

            plt.pause(.03)
            toc = time.time()
        print('Finished drawing')
    else:
        for j, m in enumerate(masses[0]):
            ax.plot(coords[j], coords[n + j], label=f'm = {m}')
        ax.legend()

    ax.set_title('Trajectories')


def draw_stats(masses, times, coords, *, axs=None, fig=None):
    '''
    Calculate & plot the total energy and angular momenta for all times.
    '''
    if axs is None or fig is None:
        fig, (momenta_ax, energy_ax) = plt.subplots(ncols=2)
    else:
        momenta_ax, energy_ax = axs
    momenta_ax.margins(x=0)
    energy_ax.margins(x=0)

    (x, y, z), (vx, vy, vz) = get_vars_from_state(coords)

    coords = np.array([x, y, z])
    momenta = masses.T * np.array([vx, vy, vz])

    L = np.cross(coords, momenta, axis=0)

    # Calculate angular momenta as L = r x p where r, p are 3D arrays
    tot_ang_mom = angular_momenta = np.sqrt(np.sum(np.cross(coords,
                                                            momenta,
                                                            axis=0)**2,
                                                   axis=0))

    T = .5 * masses.T * (vx**2 + vy**2 + vz**2)
    U = np.zeros([masses.size, times.size])
    for j in range(len(times)):
        x_separation = x[:, j] - x[:, j].reshape(-1, 1)
        y_separation = y[:, j] - y[:, j].reshape(-1, 1)
        z_separation = z[:, j] - z[:, j].reshape(-1, 1)

        np.fill_diagonal(x_separation, np.nan)
        np.fill_diagonal(y_separation, np.nan)
        np.fill_diagonal(z_separation, np.nan)

        Us = (masses * masses.T
              / np.sqrt(x_separation**2 + y_separation**2 + z_separation**2))
        np.fill_diagonal(Us, 0)

        U[:, j] = -GRAVITY * np.sum(Us, axis=0)

    for m, *L in np.column_stack((masses[0], tot_ang_mom)):
        momenta_ax.plot(times, L, label=f'm = {m}')
    momenta_ax.plot(times, np.sum(tot_ang_mom, axis=0), label='Total')

    for m, *E in np.column_stack((masses[0], U + T)):
        energy_ax.plot(times, E, label=f'm = {m}')
    energy_ax.plot(times, np.sum(U, axis=0), label='Total')

    momenta_ax.legend()
    energy_ax.legend()
    momenta_ax.set_title('Angular Momentum')
    energy_ax.set_title('Total Energy')


def solve_for(file_name):
    '''Solves the N-body problem for initial values in json file.'''
    # Vectorise the initial values and extract parameters
    masses, initial_values, tf, tmax = load_bodies_from_json(file_name)

    hash_ = get_hash(initial_values, tf, tmax)

    # Load the trajectory from file if it has already been solved
    # Otherwise solve with initial conditions and write to file
    if hash_ in parse_index(INDEX_FILE):
        print('Loading trajectories from file...')
        times, coords = load_data(DATA_DIR, hash_)
    else:
        print('Calculating trajectories...')

        d = partial(derivatives, masses)

        tic = time.time()
        sol = solve_ivp(d, [0, tf], initial_values, max_step=tmax)
        toc = time.time()
        print(f'Solved in {toc - tic:g}')

        times = sol.t
        coords = sol.y

        num_points = FRAMERATE * tf
        step = len(sol.t) // num_points

        create_data_dir(DATA_DIR)
        write_data(hash_, DATA_DIR, times, coords)
        update_index(INDEX_FILE, hash_)

    return masses, times, coords, tf


if __name__ == '__main__':
    # file_name = input('Initial values file name: ')
    file_name = FILE_NAME
    masses, times, coords, tf = solve_for(file_name)

    # draw_bodies(masses, times, coords, tf=tf, animate=True)
    draw_stats(masses, times, coords)

    plt.show()
