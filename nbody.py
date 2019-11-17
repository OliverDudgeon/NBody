'''
Numerical solutions to the 2D N-body problem
'''

from typing import List
import json

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
GRAVITY = 1


# Functions


def derivatives(masses: np.ndarray, time: float, state: List[float]):
    '''
    * @param masses
    * @param time current time value
    * @param state vectorised list of variables
    '''
    n = len(state) // 4

    # Nested lists to force numpy to create 2D array so row vector can
    # be transposed into a column vector
    x = np.array([state[:n]])
    y = np.array([state[n:2*n]])

    vx = np.array(state[2*n:3*n])
    vy = np.array(state[3*n:4*n])

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


if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.margins(x=0, y=0)

    with open('bodies.json', 'r') as bodies_handler:
        initial_values = json.loads(bodies_handler.read())

        masses = np.array([[body['m'] for body in initial_values]])
        x = [body['x'] for body in initial_values]
        y = [body['y'] for body in initial_values]
        v0x = [body['v0x'] for body in initial_values]
        v0y = [body['v0y'] for body in initial_values]

        vectorised_initial_values = np.concatenate((x, y, v0x, v0y))

    d = partial(derivatives, masses)

    sol = solve_ivp(d, [0, 20], vectorised_initial_values, max_step=1)

    # ax.plot(sol.y[0], sol.y[3])
    # ax.plot(sol.y[1], sol.y[4])
    # ax.plot(sol.y[2], sol.y[5])

    for i, t in enumerate(sol.t):
        ax.clear()
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        n = len(initial_values)
        for j in range(n):
            ax.plot(sol.y[j][i], sol.y[j + n][i], 'o')

        plt.pause(.001)

    # plt.show()
