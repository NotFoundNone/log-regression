import numpy as np

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = [list(map(float, line.strip().split(','))) for line in lines]
    return np.array(data)
