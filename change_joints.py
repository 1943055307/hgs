import json
import torch
from tqdm import tqdm

def main():
    json_file_path = 'transformation_matrices_data.json'
    inv_json_file_path = 'inv_matrix_jump.json'
    output_file_path = 'canonical_joints_jump.json'
    output_data = {}

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    with open(inv_json_file_path, 'r') as inv_json_file:
        inv_data = json.load(inv_json_file)

    for pic_name in tqdm(data.keys(), desc="Processing"):
        transformation_matrices = torch.tensor(data[pic_name]['transformation_matrices'], dtype=torch.float32)  # Nx4x4
        Rt = torch.tensor(inv_data[pic_name]['Rt'], dtype=torch.float32)  # 4x4

        # 初始化存储规范关节的转换矩阵
        canonical_transformation_matrices = []

        for i in range(transformation_matrices.shape[0]):
            transformed_joint_matrix = transformation_matrices[i]
            canonical_joint_matrix = Rt @ transformed_joint_matrix
            canonical_transformation_matrices.append(canonical_joint_matrix.numpy().tolist())

        output_data[pic_name] = {
            'canonical_transformation_matrices': canonical_transformation_matrices
        }

    with open(output_file_path, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

if __name__ == "__main__":
    main()
