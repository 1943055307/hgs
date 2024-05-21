import json
import numpy as np
from plyfile import PlyData, PlyElement

def storePly(path, xyz, normals, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    elements = np.zeros(xyz.shape[0], dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    elements['red'] = rgb[:, 0]
    elements['green'] = rgb[:, 1]
    elements['blue'] = rgb[:, 2]

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(path)
    
def save_point_cloud(vertices, filename):
    xyz = np.array(vertices)
    num_pts = xyz.shape[0]
    normals = np.zeros_like(xyz)
    C0 = 0.28209479177387814
    shs = np.random.random((num_pts, 3)) / 255.0
    rgb = (shs * C0 + 0.5) * 255

    storePly(filename, xyz, normals, rgb)

def main():
    json_file_path = 'canonical_joints_jump.json'
    output_dir = 'joints_ply'
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    for pic_name, joints_data in data.items():
        transformation_matrices = np.array(joints_data['canonical_transformation_matrices'])
        
        joints = transformation_matrices[:, :3, 3]
        
        output_file_path = f"{output_dir}/{pic_name}.ply"
        
        save_point_cloud(joints, output_file_path)

if __name__ == "__main__":
    main()
