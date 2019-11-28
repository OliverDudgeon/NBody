'''Useful development helper functions'''

import numpy as np


def pprint(x, *, prec=3, supsm=True):
    '''Pretty print nparray for inspection'''
    print(np.array_repr(x, max_line_width=100,
                        precision=prec, suppress_small=supsm))
