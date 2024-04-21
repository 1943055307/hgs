import json
import numpy as np
import torch
import pickle
import open3d as o3d
from tqdm import tqdm
from smplpytorch.pytorch.smpl_layer import SMPL_Layer

def apply_transformation(joints, transform_matrix):
    joints_torch = torch.tensor(joints, dtype=torch.float32)
    transformed_joints = torch.mm(transform_matrix[:3, :3], joints_torch.T).T + transform_matrix[:3, 3]
    return transformed_joints.numpy()


def main():
    json_file_path = 'transformed_joints_data.json'
    inv_json_file_path = 'inv_matrix.json'
    output_file_path = 'canonical_pose_joints.json'
    output_data = {}

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    with open(inv_json_file_path, 'r') as inv_json_file:
        inv_data = json.load(inv_json_file)

    for pic_name in data.keys():
        transformed_joints = data[pic_name]['transformed_joints']
        R = torch.tensor(inv_data[pic_name]['R'], dtype=torch.float32)
        T = torch.tensor(inv_data[pic_name]['T'], dtype=torch.float32)

        Rt = torch.zeros((4, 4), dtype=torch.float32)  
        Rt[:3, :3] = R.transpose(0, 1)
        Rt[:3, 3] = T
        Rt[3, 3] = 1.0

        transformed_joints = apply_transformation(transformed_joints, torch.tensor(Rt).transpose(0, 1))
        output_data[pic_name] = transformed_joints.tolist()

    with open(output_file_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

if __name__ == "__main__":
    main()

