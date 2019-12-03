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
    * @param state Array of corrdinates in vector form. Length must be
        integer multiples of m.

    Returns Tuple of a positions list values and a speeds list
    '''
    coords = np.split(state, 2*d)
    return coords[:d], coords[d:2*d]


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

    masses = np.array([body['m'] for body in initial_values], ndmin=2)

    coord_names = ['x', 'y', 'z', 'v0x', 'v0y', 'v0z']

    initial_state = [[body[name] for body in initial_values]
                     for name in coord_names]

    return masses, np.concatenate(initial_state), dump['tf'], dump['tmax']


def draw_bodies(masses, times, coords, *,
                animate=False, tf=None, ax=None, fig=None):
    '''
    Plot trajectories of the bodies.
    * @param masses numpy array of the masses
    * @param times numpy array of time values
    * @param coords numpy array of vectorised coordinates
        - rows are the coordinate index
        - columns are the time index
    * @param animate whether to animate the trajectories
    * @param tf duration of animation
    * @param ax axis to plot onto
    * @param fig figure window to redraw
    '''

    x, y, z, *_ = np.split(coords, 2*d)

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
            ax.set_xlim(np.min(x), np.max(x))
            ax.set_ylim(np.min(y), np.max(y))

            idx = int(len(times) * (toc - tic) / tf)
            for j in range(n):
                l, = ax.plot(x[j][:idx], y[j][:idx])
                ax.plot(coords[j][idx], coords[n + j][idx], marker='o',
                        c=l.get_c())

            plt.pause(.03)
            toc = time.time()
        print('Finished drawing')
    else:
        for j, m in enumerate(masses[0]):
            ax.plot(coords[j], coords[n + j], label=f'm = {m}')
        ax.legend()

    ax.set_title('Trajectories')


def draw_stats(masses, times, state, *, rel_L=True, rel_E=True,
               axs=None, fig=None):
    '''
    Calculate & plot the total energy and angular momenta for all times.
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

    n = masses.size

    x, y, z, vx, vy, vz = np.split(state, 2*d)
    r = np.array([x, y, z])
    speeds = np.array([vx, vy, vz])

    Ls = np.cross(r, masses.T * speeds, axis=0)
    L_tot = np.sum(Ls, axis=1)

    T = .5 * masses.T * np.sum(speeds**2, axis=0)

    U = np.zeros([n, len(times)])
    for j in range(len(times)):
        sep = [pos[:, j] - pos[:, j].reshape(-1, 1) for pos in r]
        for s in sep:
            np.fill_diagonal(s, np.inf)

        Us = masses * masses.T / np.sqrt(sum(s**2 for s in sep))
        U[:, j] = -GRAVITY * np.sum(np.triu(Us), axis=0)

    L = np.sqrt(np.sum(L_tot**2, axis=0))
    if rel_L:
        momenta_ax.plot(times, L / L[0] - 1, label='$L / L_0 - 1$')
    else:
        momenta_ax.plot(times, L, label='$L$')

    E = np.sum(U + T, axis=0)
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
    * @param file_name name of file without extension to get initial value from
    * @param Just (re)calculate even if there is a cashed version
    '''
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

    draw_bodies(masses, times, coords, tf=tf, animate=True)
    # draw_stats(masses, times, coords, rel_L=False)

    plt.show()
