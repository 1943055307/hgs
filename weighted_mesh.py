import json
import numpy as np

weights = [0.012, 0.032, 0.086, 0.234, 0.636]

with open('canonical_mesh_jump.json', 'r') as f:
    data = json.load(f)

mesh_keys = list(data.keys())

for key in mesh_keys:
    data[key] = np.array(data[key])

processed_data = {}
for i in range(len(mesh_keys)):
    if i < 4:
        processed_data[mesh_keys[i]] = data[mesh_keys[i]].tolist()
    else:
        weighted_vertices = np.zeros_like(data[mesh_keys[i]])
        for j in range(5):
            weighted_vertices += data[mesh_keys[i - 4 + j]] * weights[j]
        processed_data[mesh_keys[i]] = weighted_vertices.tolist()

with open('canonical_mesh_jump_weighted.json', 'w') as f:
    json.dump(processed_data, f, indent=2)

print("Saved In canonical_mesh_jump_weighted.json ")
