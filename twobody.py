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
    x1_speed, x1, x2_speed, x2, y1_speed, y1, y2_speed, y2 = state

    x_separation = x1 - x2
    y_separation = y1 - y2
    cubed_distance = ((x_separation)**2 + (y_separation)**2)**1.5

    T = .5*(mass1 * (x1_speed**2 + y1_speed**2) + mass2 * (x2_speed**2 + y2_speed**2))
    V = -gravity * mass1 * mass2 / np.sqrt(x_separation**2 + y_separation**2)


    # print(T + V)

    x_force = -gravity * x_separation / cubed_distance
    y_force = -gravity * mass2 * y_separation / cubed_distance
    return (mass2 * x_force, x1_speed, -mass1 * x_force, x2_speed,
            mass2*y_force, y1_speed, -mass1 * y_force, y2_speed)


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

    sol = solve_ivp(derivatives, [0, 5], list(init.values()),
                    max_step=.001)

    # ax.plot(sol.y[1], sol.y[5])
    # ax.plot(sol.y[3], sol.y[7])

    # plt.show()

    for i, t in enumerate(sol.t):
        if not i % 100:
            ax.clear()
            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)
            ax.plot(sol.y[1][i], sol.y[5][i], 'o')
            ax.plot(sol.y[3][i], sol.y[7][i], 'o')

            plt.pause(.001)

