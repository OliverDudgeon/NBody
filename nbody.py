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

# Constants
FILE_NAME = 'figureeight'

DATA_DIR = 'data'
INDEX_FILE = 'index'

GRAVITY = 1

m = 4  # Number of coordinates (2 position 2 speed)
FRAMERATE = 60  # Number of data points to be saved per unit time


# Functions


def get_vars_from_state(state):
    '''
    Slice state into x position, y position, x speed and y speed.
    * @param state List of corrdinates in vector form. Length must be
        integer multiples of 4.
    '''
    n = len(state) // m

    # 2D arrays required for transposing
    x = np.array(state[:n], ndmin=2)
    y = np.array(state[n:2*n], ndmin=2)

    vx = np.array(state[2*n:3*n])
    vy = np.array(state[3*n:4*n])

    return x, y, vx, vy


def derivatives(masses, time, state):
    '''
    Calculate the values of the speed and acceleration of each body.
    * @param masses
    * @param time current time value
    * @param state vectorised list of variables

    Returns Derivatives in vectorised form
    '''
    x, y, vx, vy = get_vars_from_state(state)

    x_separation = x - x.T
    y_separation = y - y.T

    cubed_separation = (x_separation**2 + y_separation**2)**1.5
    np.fill_diagonal(cubed_separation, np.nan)

    x_acceleration = -GRAVITY * masses.T * x_separation / cubed_separation
    y_acceleration = -GRAVITY * masses.T * y_separation / cubed_separation

    np.fill_diagonal(x_acceleration, 0)
    np.fill_diagonal(y_acceleration, 0)

    return np.concatenate((vx, vy, np.sum(x_acceleration, axis=0),
                           np.sum(y_acceleration, axis=0)))


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
        v0x = [body['v0x'] for body in initial_values]
        v0y = [body['v0y'] for body in initial_values]

        return (masses, np.concatenate((x, y, v0x, v0y)),
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


def draw_stats(masses, times, coords):
    '''
    Calculate & plot the total energy and angular momenta for all times.
    '''
    fig, (momenta_ax, energy_ax) = plt.subplots(ncols=2)
    momenta_ax.margins(x=0)
    energy_ax.margins(x=0)

    x, y, vx, vy = get_vars_from_state(coords)

    angular_momenta = masses.T * (y*vx - x*vy)
    T = .5 * masses.T * (vx**2 + vy**2)
    U = np.zeros([masses.size, times.size])
    for j in range(len(times)):
        x_separation = x[:, j] - x[:, j].reshape(-1, 1)
        y_separation = y[:, j] - y[:, j].reshape(-1, 1)

        np.fill_diagonal(x_separation, np.nan)
        np.fill_diagonal(y_separation, np.nan)

        Us = (masses * masses.T
              / np.sqrt(x_separation**2 + y_separation**2))
        np.fill_diagonal(Us, 0)

        U[:, j] = -GRAVITY * np.sum(Us, axis=0)

    for m, *L in np.column_stack((masses[0], angular_momenta)):
        momenta_ax.plot(times, L, label=f'm = {m}')
    momenta_ax.plot(times, np.sum(angular_momenta, axis=0), label='Total')

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

    draw_bodies(masses, times, coords, tf=tf, animate=True)
    draw_stats(masses, times, coords)

    plt.show()
