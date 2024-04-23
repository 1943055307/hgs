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
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        # Get the base filename without extension
        file_name = os.path.basename(frame['file_path'])
        if file_name.endswith('.png'):
            file_name = file_name[:-4]  # Remove ".png" from the file name
        
        inv_matrices[file_name] = {
            "R": R.tolist(),
            "T": T.tolist() 
        }

    with open(output_filepath, 'w') as outfile:
        json.dump(inv_matrices, outfile, indent=4)

if __name__ == "__main__":
    input_filepath = 'data/tshirt/transforms_train.json'
    output_filepath = 'inv_matrix.json'
    invert_matrices(input_filepath, output_filepath)
