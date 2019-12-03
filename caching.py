'''Utility functions for caching N-body results.'''

import os
import json
from hashlib import sha256
import numpy as np


def create_index(index_file):
    '''Create index file if it doesn't already exist.'''
    if not os.path.exists(f'{index_file}.json'):
        with open(f'{index_file}.json', 'w'):
            pass


def create_data_dir(data_dir):
    '''Create data directory if it doesn't exist.'''
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


def parse_index(index_file):
    '''Parse and return the index file.'''
    create_index(index_file)
    with open('index.json') as index_handler:
        dump_str = index_handler.read()
        if dump_str == '':
            return []
        else:
            return json.loads(dump_str)


def get_hash(*args):
    '''
    Calc hashed representation of initial values.
    In order to identify if the initial conditions / parameters have
    already been used.
    '''
    hasher = sha256()

    dat = bytes(''.join(repr(a) for a in args), encoding='utf8')
    hasher.update(dat)
    return hasher.hexdigest()


def load_data(data_dir, hash_):
    '''Load data files into numpy arrays.'''
    with open(f'{data_dir}/{hash_}_t') as sol_file_handler:
        times = np.loadtxt(sol_file_handler)
    with open(f'{data_dir}/{hash_}_y') as sol_file_handler:
        coords = np.loadtxt(sol_file_handler)
    return times, coords


def write_data(hash_, data_dir, times, coords, *, prec=8):
    '''Write times and coords data to data dir.'''
    print(f'Writing times data to data/{hash_[:15]}...')
    with open(f'{data_dir}/{hash_}_t', 'w+') as sol_file_handler:
        np.savetxt(sol_file_handler, times, fmt=f'%.{prec}f')
    print('Finished writing times data')

    print(f'Writing coordinates to data/{hash_[:15]}...')
    with open(f'{data_dir}/{hash_}_y', 'w+') as sol_file_handler:
        np.savetxt(sol_file_handler, coords, fmt=f'%.{prec}f')
    print('Finished writing coordinates data')


def update_index(index_file, hash_):
    '''Update index file with new hash.'''
    with open(f'{index_file}.json') as index_handler:
        dump_str = index_handler.read()
    if dump_str == '':
        index = [hash_]
    else:
        index = json.loads(dump_str)
        index.append(hash_)
    with open(f'{index_file}.json', 'w+') as index_handler:
        json.dump(index, index_handler)
    print('Updated index')
