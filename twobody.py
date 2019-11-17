'''
2D 2-body problem
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

gravity, mass1, mass2 = 1, 100, 1


def derivatives(time, state):
    '''
    '''
    x1, x2, y1, y2, x1_speed, x2_speed, y1_speed, y2_speed = state

    x_separation = x1 - x2
    y_separation = y1 - y2
    cubed_distance = ((x_separation)**2 + (y_separation)**2)**1.5

    x_force = -gravity * x_separation / cubed_distance
    y_force = -gravity * y_separation / cubed_distance
    return (x1_speed, x2_speed, y1_speed, y2_speed, mass2 * x_force,
            -mass1 * x_force, mass2*y_force, -mass1 * y_force)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.margins(x=0, y=0)

    init = {
        'x1_speed': -.05,
        'x1': 0,
        'x2_speed': 7,
        'x2': 0,
        'y1_speed': 0,
        'y1': 0,
        'y2_speed': 0,
        'y2': -2
    }
    sol = solve_ivp(derivatives, [0, 5], [0, 0, 0, -2, -.05, 7, 0, 0],
                    max_step=.001)

    # ax.plot(sol.y[0], sol.y[2])
    # ax.plot(sol.y[1], sol.y[3])

    # plt.show()

    for i, t in enumerate(sol.t):
        if not i % 100:
            ax.clear()
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.plot(sol.y[0][i], sol.y[2][i], 'o')
            ax.plot(sol.y[1][i], sol.y[3][i], 'o')

            plt.pause(.001)

