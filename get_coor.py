import json
import numpy as np
import torch
from tqdm import tqdm
from smplpytorch.pytorch.smpl_layer import SMPL_Layer

def apply_transformation(joints, transform_matrix):
    joints_torch = torch.tensor(joints, dtype=torch.float32)
    transformed_joints = torch.mm(transform_matrix[:3, :3], joints_torch.T).T + transform_matrix[:3, 3]
    return transformed_joints.numpy()

def main():
    json_file_path = 'keypoints_data_s.json'
    output_data = {}

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

        for key, value in tqdm(data.items(), desc="processing"):
            joint_coordinates = np.array(value['pred_xyz_jts_24_struct']).reshape(24, 3)
            # transformed_joints = apply_transformation(joint_coordinates, transform_matrix)
            # If you want to use transformed joints, uncomment the line above and use transformed_joints instead of joint_coordinates below.

            # Store the coordinates as a list of lists directly in the dictionary under the key
            output_data[key] = {'coordinates': joint_coordinates.tolist()}

    with open('transformed_coor_data.json', 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

if __name__ == "__main__":
    main()
