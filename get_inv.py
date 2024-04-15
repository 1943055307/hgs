import json
import numpy as np
import os
from tqdm import tqdm

def invert_matrices(input_filepath, output_filepath):
    # Read the input JSON file
    with open(input_filepath, 'r') as file:
        data = json.load(file)["frames"]
    
    inv_matrices = {}

    for frame in tqdm(data, desc="Calculating Inverse Matrices"):
        matrix = np.array(frame['transform_matrix'])
        inv_matrix = np.linalg.inv(matrix)
        inv_matrices[os.path.basename(frame['file_path'])] = {
            "inv_matrix": inv_matrix.tolist() 
        }

    with open(output_filepath, 'w') as outfile:
        json.dump(inv_matrices, outfile, indent=4)

if __name__ == "__main__":
    input_filepath = 'data/transforms_train.json'
    output_filepath = 'inv_matrix.json'
    invert_matrices(input_filepath, output_filepath)
