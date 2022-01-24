import os.path

import scipy.io as io
import numpy as np
from tepimport import add_lagged_samples

_folder_path = os.path.abspath("./CVACaseStudy/CVACaseStudy/")
FILE_NAMES = (
    ('Training Data', 'Training.mat'),
    ('Faulty Case 1', 'FaultyCase1.mat'),
    ('Faulty Case 2', 'FaultyCase2.mat'),
    ('Faulty Case 3', 'FaultyCase3.mat'),
    ('Faulty Case 4', 'FaultyCase4.mat'),
    ('Faulty Case 5', 'FaultyCase5.mat'),
    ('Faulty Case 6', 'FaultyCase6.mat'),
)
FAULT_START_STOP = (
    ((0, 0), (0, 0), (0, 0)),
    ((1565, 5180), (656, 3776), (690, 3690)),
    ((2243, 6615), (475, 2655), (330, 2466)),
    ((1135, 8351), (332, 5870), (595, 9565)),
    ((952, 6293), (850, 3850), (240, 3240)),
    ((685, 1171, 1171, 2252), (1632, 2954, 7030, 7552, 8056, 10607)),
    ((1722, 2799), (1036, 4829))
)


def set_folder_path(path: str) -> None:
    global _folder_path
    abs_path = os.path.abspath(path)
    if not os.path.isdir(abs_path):
        raise NotADirectoryError
    _folder_path = abs_path


def import_data_set(file_name: str) -> list:
    file_path = os.path.join(_folder_path, file_name)
    file_data = io.loadmat(file_path)
    # Don't import these sets
    ignored_sets = ['__header__', '__version__', '__globals__']
    data = [[None, None],
            [None, None],
            [None, None]]

    for key, item in file_data.items():
        if key in ignored_sets:
            continue
        if item.shape[0] > item.shape[1]:
            # Enfore column samples and row variables
            item = item.T
        set_number = int(key[-1])
        if 'EvoFault' in key:
            data[set_number - 1][1] = item
        elif ('Set' in key) or (key == f'T{set_number}'):
            data[set_number - 1][0] = item
    return(data)


def import_sets(lagged_samples: int = 0) -> list:
    """
    Returns a list of all the data sets in the form of
    Set name, Set data, Fault range, Fault evolution
    """
    data_sets = []
    for i, (name, file_name) in enumerate(FILE_NAMES):
        data = import_data_set(file_name)
        if i > 4:  # Sets 5 and 6 have only 2 data sets
            data.pop()
        for j, (X, F) in enumerate(data):
            F_rng = [s for s in FAULT_START_STOP[i][j]]
            if lagged_samples > 0:
                X = add_lagged_samples(X, lagged_samples)
                if F is not None:
                    F = F[lagged_samples:]
                F_rng = [max(s - lagged_samples, 0) for s in F_rng]
            data_sets.append((f'{name}.{j + 1}', X, F_rng, F))
    return(data_sets)


if __name__ == "__main__":
    print("This file cannot be run directly, import this module to obtain the",
          "datasets of the Multiphase Flow Facility process")
