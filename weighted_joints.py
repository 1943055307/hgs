import json
import numpy as np

weights = [0.012, 0.032, 0.086, 0.234, 0.636]

with open('canonical_joints_jump.json', 'r') as f:
    data = json.load(f)

mesh_keys = list(data.keys())

# 将数据转换为 numpy 数组
for key in mesh_keys:
    data[key]['canonical_transformation_matrices'] = np.array(data[key]['canonical_transformation_matrices'])

processed_data = {}
for i in range(len(mesh_keys)):
    if i < 4:
        processed_data[mesh_keys[i]] = {
            "weighted_transformation_matrices": data[mesh_keys[i]]['canonical_transformation_matrices'].tolist()
        }
    else:
        weighted_matrices = np.zeros_like(data[mesh_keys[i]]['canonical_transformation_matrices'])
        for j in range(5):
            weighted_matrices += data[mesh_keys[i - 4 + j]]['canonical_transformation_matrices'] * weights[j]
        processed_data[mesh_keys[i]] = {
            "weighted_transformation_matrices": weighted_matrices.tolist()
        }

with open('canonical_joints_jump_weighted.json', 'w') as f:
    json.dump(processed_data, f, indent=2)

print("Saved in canonical_joints_jump_weighted.json")
