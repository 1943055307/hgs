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
    # 定义向量
    vec = np.array([-0.02128927, 0.07789805, -2.5382574 ]) * (-1.0 / 4.005270504951477)
    # vec = np.array([0, 0, 0 ])
    vec = vec + np.array([-0.06, 0.06, 0.1 ])
    translation_vector = torch.tensor( vec,  dtype=torch.float32)
    transformed_joints += translation_vector
    return transformed_joints.numpy()




def main():
    json_file_path = 'transformed_pose_data.json'
    inv_json_file_path = 'inv_matrix.json'
    output_file_path = 'canonical_mesh.json'
    translation_file_path = 'keypoints_data.json'
    output_data = {}

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    with open(inv_json_file_path, 'r') as inv_json_file:
        inv_data = json.load(inv_json_file)

    with open(translation_file_path, 'r') as trans_json_file:
        trans_data = json.load(trans_json_file)

    for pic_name in data.keys():
        transformed_joints = data[pic_name]['transformed_vertices']
        Rt = torch.tensor(inv_data[pic_name]['Rt'], dtype=torch.float32)

        transformed_joints = apply_transformation(transformed_joints, torch.tensor(Rt))
        output_data[pic_name] = transformed_joints.tolist()

    with open(output_file_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

if __name__ == "__main__":
    main()

