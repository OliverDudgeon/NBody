'''
Functions to calculate physical values related to the N-body problem.
'''

import numpy as np


def get_vars_from_state(d, state):
    '''
    Slice state into coords and speeds.
    Coords and speed are split into x, y, z components within the lists.
    * @param state Array of corrdinates in vector form. Length must be
        integer multiples of m.

    Returns Tuple of a positions list values and a speeds list
    '''
    coords = np.split(state, 2*d)
    return coords[:d], coords[d:2*d]


def derivatives(d, gravity, masses, time, state):
    '''
    Calculate the values of the speed and acceleration of each body.
    * @param gravity Universal gravitational constant is chosen units
    * @param masses
    * @param time current time value
    * @param state vectorised list of variables

    Returns Derivatives in vectorised form
    '''
    coords, speeds = get_vars_from_state(d, state)

    # q - q.T computes separations as a matrix if q is the 1D array
    # of position values for the bodies (for a given dimension).
    separations = np.array([q - np.array([q]).T for q in coords])

    cubed_separation = np.sum(separations**2, axis=0)**1.5

    # Replace the zeros with inf to avoid 0/0 error and so force on a
    # particle due to itself is zero.
    np.fill_diagonal(cubed_separation, np.inf)

    acceleration = -gravity * np.sum(masses.T
                                     * separations
                                     / cubed_separation, axis=1)

    return np.concatenate([*speeds,
                           acceleration.reshape(acceleration.size)])


def split_state_into_coords(state, d):
    x, y, z, vx, vy, vz = np.split(state, 2*d)
    r = np.array([x, y, z])
    speeds = np.array([vx, vy, vz])

    return r, speeds


def calc_total_energy(gravity, d, masses, state):
    '''
    Calculate the total energy of a N-body system.
    Adds up the kinetic energy of each particle along with the
    gravitational potential energy of each pair.
    * @param gravity Universal gravitation constant in chosen units
    * @param d Number of dimensions in model
    * @params masses 1D numpy array of masses
    * @params state 2D array of position and speeds values at each time

    Returns 1D array of total energy values at each time
    '''

    r, speeds = split_state_into_coords(state, d)
    _, num_tvals = state.shape  # Length of state and times is equal

    T = .5 * masses.T * np.sum(speeds**2, axis=0)

    U = np.zeros([masses.size, num_tvals])
    for j in range(num_tvals):
        # Reshape 1D array into 2D column vector
        sep = [pos[:, j] - pos[:, j].reshape(-1, 1) for pos in r]
        for s in sep:
            # Stop 0/0 errors
            np.fill_diagonal(s, np.inf)

        Us = masses * masses.T / np.sqrt(sum(s**2 for s in sep))
        U[:, j] = -gravity * np.sum(np.triu(Us), axis=0)

    return np.sum(U + T, axis=0)


def calc_total_ang_momentum(d, masses, state):
    '''
    Calculate the total angular momentum of the system.
    * @param d Number of dimensions in model
    * @params masses 1D numpy array of masses
    * @params state 2D array of position and speeds values at each time

    Returns 1D array of the magnitude of the vector sum of all the
    individual angular momentum vectors at each time value
    '''
    x, y, z, vx, vy, vz = np.split(state, 2*d)
    r = np.array([x, y, z])
    speeds = np.array([vx, vy, vz])

    # Angular momentum = r x mv
    Ls = np.cross(r, masses.T * speeds, axis=0)
    L_tot = np.sum(Ls, axis=1)

    return np.sqrt(np.sum(L_tot**2, axis=0))
