import json
import numpy as np
import torch
from tqdm import tqdm
from smplpytorch.pytorch.smpl_layer import SMPL_Layer

def rotation_matrix2axis_angle(R):
    theta = torch.acos(torch.clamp((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2, -1.0, 1.0))
    axis = torch.stack([(R[:, 2, 1] - R[:, 1, 2]),
                        (R[:, 0, 2] - R[:, 2, 0]),
                        (R[:, 1, 0] - R[:, 0, 1])], dim=1) # rot axis
    sin_theta = torch.sin(theta)
    axis = torch.where(sin_theta.unsqueeze(-1) >= 1e-7, axis / (2 * sin_theta).unsqueeze(-1), axis)
    axis_angle = axis * theta.unsqueeze(-1)
    return axis_angle

def apply_transformation(vertices, joints, transform_matrix):
    vertices_torch = torch.tensor(vertices, dtype=torch.float32)
    joints_torch = torch.tensor(joints, dtype=torch.float32)
    transformed_vertices = torch.mm(transform_matrix[:3, :3], vertices_torch.T).T + transform_matrix[:3, 3]
    transformed_joints = torch.mm(transform_matrix[:3, :3], joints_torch.T).T + transform_matrix[:3, 3]
    return transformed_vertices.numpy(), transformed_joints.numpy()

def main():

    smpl_layer = SMPL_Layer(
        gender='male',
        model_root='smpl/models'
    )

    json_file_path = 'HybrIK/SMPL/keypoints_data.json'
    output_data = {}
    output_joints = {}

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

        for key, value in tqdm(data.items(), desc="processing"):
            theta_params = torch.tensor(value['pred_theta_mats'], dtype=torch.float32)
            rotation_matrices = theta_params.view(24, 3, 3)
            axis_angles = torch.zeros(24, 3)
            for i in range(24):
                axis_angle = rotation_matrix2axis_angle(rotation_matrices[i].unsqueeze(0))
                axis_angles[i] = axis_angle.squeeze(0)

            pose_params = axis_angles.view(1, 72)
            shape_params = torch.tensor(value['pred_shape'], dtype=torch.float32)
            vertices, joints = smpl_layer(pose_params, shape_params)

            transform_matrix = torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ])
            transformed_vertices, transformed_joints = apply_transformation(vertices.detach().numpy().squeeze(), joints.detach().numpy().squeeze(), transform_matrix)

            output_data[key] = {
                'transformed_joints': transformed_joints.tolist(),
                'transformed_vertices': transformed_vertices.tolist()
            }

            output_joints[key] = {
                'transformed_joints': transformed_joints.tolist()
            }

    with open('transformed_pose_data.json', 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    with open('transformed_joints_data.json', 'w') as outfile:
        json.dump(output_joints, outfile, indent=4)

if __name__ == "__main__":
    main()
