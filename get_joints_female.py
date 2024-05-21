import json
import numpy as np
import torch
from tqdm import tqdm
from smplpytorch.pytorch.smpl_layer import SMPL_Layer

def rotation_matrix2axis_angle(R):
    theta = torch.acos(torch.clamp((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2, -1.0, 1.0))
    axis = torch.stack([(R[:, 2, 1] - R[:, 1, 2]),
                        (R[:, 0, 2] - R[:, 2, 0]),
                        (R[:, 1, 0] - R[:, 0, 1])], dim=1)  # rot axis
    sin_theta = torch.sin(theta)
    axis = torch.where(sin_theta.unsqueeze(-1) >= 1e-7, axis / (2 * sin_theta).unsqueeze(-1), axis)
    axis_angle = axis * theta.unsqueeze(-1)
    return axis_angle

def create_transformation_matrix(rotation_matrix, translation_vector):
    transformation_matrix = torch.eye(4, dtype=torch.float32)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = torch.tensor(translation_vector, dtype=torch.float32)  # 转换 translation_vector 为 torch.Tensor
    return transformation_matrix

def main():
    smpl_layer = SMPL_Layer(
        gender='female',
        model_root='smpl/models'
    )

    json_file_path = 'HybrIK/SMPL/keypoints_data.json'
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

            joints = joints.detach().numpy().squeeze()

            transformation_matrices = []
            for i in range(24):
                rotation_matrix = rotation_matrices[i]
                translation_vector = joints[i]
                transformation_matrix = create_transformation_matrix(rotation_matrix, translation_vector)
                transformation_matrices.append(transformation_matrix.numpy().tolist())

            output_joints[key] = {
                'transformation_matrices': transformation_matrices
            }

    with open('transformation_matrices_data.json', 'w') as outfile:
        json.dump(output_joints, outfile, indent=4)

if __name__ == "__main__":
    main()
