import numpy as np

def load_data(filename, shape):
    """Load matrix data from a file."""
    with open(filename, 'r') as file:
        return np.fromfile(file, sep=' ').reshape(shape)
