'''
Numerical solutions to the 2D N-body problem
'''

import os
import json
import time
from hashlib import sha256
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
DATA_DIR = 'data'
INDEX_FILE = 'index'
GRAVITY = 1
m = 4  # Number of coordinates (2 position 2 speed)
FRAMERATE = 60

file_name = 'pythag'

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


def draw_bodies(masses, times, coords, *, tf, animate=False):
    '''
    Plot trajectories of the bodies.
    * @param m asses numpy array of time values
    * @param coords numpy array of vectorised coordinates
        - rows are the coordinate index
        - columns are the time index
    * @param animate whether to animate the trajectories
    '''
    fig, ax = plt.subplots()
    ax.margins(x=0, y=0)

    n = np.size(masses)

    if animate:
        print('Drawing...')

        tic = time.time()
        toc = tic
        while toc - tic < tf:
            ax.clear()
            ax.set_xlim(-8, 8)
            ax.set_ylim(-8, 8)

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


# Create index file if it doesn't already exist
if not os.path.exists(f'{INDEX_FILE}.json'):
    with open(f'{INDEX_FILE}.json', 'w'):
        pass

# Parse the index file
with open('index.json') as index_handler:
    dump_str = index_handler.read()
    if dump_str == '':
        index = []
    else:
        index = json.loads(dump_str)

# Vectorise the initial values and extract parameters
masses, initial_values, tf, tmax = load_bodies_from_json(file_name)

# Hashed representation of initial values to identify if the initial
# conditions / parameters have already been used.
hasher = sha256()
dat = bytes(repr(initial_values) + repr(tf) + repr(tmax), encoding='utf8')
hasher.update(dat)
hash_ = hasher.hexdigest()

# Load the trajectory from file if it has already been solved
# Otherwise solve with initial conditions and write to file
if hash_ in index:
    print('Loading trajectories from file')

    with open(f'data/{hash_}_t') as sol_file_handler:
        times = np.loadtxt(sol_file_handler)
    with open(f'data/{hash_}_y') as sol_file_handler:
        coords = np.loadtxt(sol_file_handler)
else:
    print('Calculating trajectories')

    d = partial(derivatives, masses)

    tic = time.time()
    sol = solve_ivp(d, [0, tf], initial_values, max_step=tmax)
    toc = time.time()
    print(f'Solved in {toc - tic:g}')

    times = sol.t
    coords = sol.y

    print(coords.shape)

    num_points = FRAMERATE * tf
    step = len(sol.t) // num_points

    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print(f'Writing times data to data/{hash_[:10]}...')
    with open(f'data/{hash_}_t', 'w+') as sol_file_handler:
        np.savetxt(sol_file_handler, sol.t[::step], fmt="%.8f")
    print('Finished writing times data')
    print(f'Writing coordinates to data/{hash_[:10]}...')
    with open(f'data/{hash_}_y', 'w+') as sol_file_handler:
        np.savetxt(sol_file_handler, sol.y[:, ::step], fmt="%.8f")
    print('Finished writing coordinates data')

    with open(f'index.json') as index_handler:
        dump_str = index_handler.read()
    if dump_str == '':
        index = [hash_]
    else:
        index = json.loads(dump_str)
        index.append(hash_)
    with open(f'index.json', 'w+') as index_handler:
        json.dump(index, index_handler)
    print('Updated index')


draw_bodies(masses, times, coords, tf=tf, animate=True)
draw_stats(masses, times, coords)

# plt.show()
