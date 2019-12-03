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

    coords = np.split(np.array(state[:len(state) // 2]), d)
    speeds = np.split(np.array(state[len(state) // 2:]), d)

    return coords, speeds


def derivatives(masses, time, state):
    '''
    Calculate the values of the speed and acceleration of each body.
    * @param masses
    * @param time current time value
    * @param state vectorised list of variables

    Returns Derivatives in vectorised form
    '''
    coords, speeds = get_vars_from_state(state)

    separations = np.array([q - np.array([q]).T for q in coords])

    cubed_separation = np.sum(separations**2, axis=0)**1.5
    np.fill_diagonal(cubed_separation, np.inf)

    acceleration = (-GRAVITY
                    * masses
                    * np.sum(separations / cubed_separation, axis=1))

    return np.concatenate([*speeds,
                           acceleration.reshape(acceleration.size)])


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

    n = masses.size

    x, y, z, vx, vy, vz = np.split(coords, 2*d)

    Lx = masses.T * (y*vz - z*vy)
    Ly = masses.T * (z*vx - y*vz)
    Lz = masses.T * (y*vx - x*vy)
    L = np.sqrt(np.sum(Lx, axis=0)**2 + np.sum(Ly, axis=0)**2 + np.sum(Ly, axis=0)**2)

    U = np.zeros([n, len(times)])
    T = .5 * masses.T * (vx**2 + vy**2 + vz**2)

    for j in range(len(times)):
        x_sep = x[:, j] - x[:, j].reshape(-1, 1)
        y_sep = y[:, j] - y[:, j].reshape(-1, 1)
        z_sep = z[:, j] - z[:, j].reshape(-1, 1)

        np.fill_diagonal(x_sep, np.inf)
        np.fill_diagonal(y_sep, np.inf)
        np.fill_diagonal(z_sep, np.inf)

        Us = (masses * masses.T
              / np.sqrt(x_sep**2 + y_sep**2 + z_sep**2))

        U[:, j] = -GRAVITY * np.sum(np.triu(Us), axis=0)

    momenta_ax.plot(times, L, label='Total')

    E = np.sum(U + T, axis=0)
    energy_ax.plot(times, E / E[0] - 1, label='$E / E_0 - 1$')

    momenta_ax.legend()
    energy_ax.legend()
    momenta_ax.set_title('Angular Momentum')
    energy_ax.set_title('Total Energy')


def solve_for(file_name, calc=False):
    '''Solves the N-body problem for initial values in json file.'''
    # Vectorise the initial values and extract parameters
    masses, initial_values, tf, tmax = load_bodies_from_json(file_name)

    hash_ = get_hash(masses, initial_values, tf, tmax)

    # Load the trajectory from file if it has already been solved
    # Otherwise solve with initial conditions and write to file
    if hash_ in parse_index(INDEX_FILE) and not calc:
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

        if not calc:
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
