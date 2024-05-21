import json
import numpy as np
import open3d as o3d
import pickle
import argparse
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

def main(chosen_index):
    o3d.visualization.gui.Application.instance.initialize()

    model_path = 'smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    with open(model_path, 'rb') as f:
        smpl_model_data = pickle.load(f, encoding='latin1')
    faces = smpl_model_data['f']

    json_file_path = 'canonical_mesh_jump_weighted.json'
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    keys = list(data.keys())
    if chosen_index < 0 or chosen_index >= len(keys):
        print(f"Invalid index {chosen_index}. Please choose an index between 0 and {len(keys) - 1}.")
        return

    chosen_key = keys[chosen_index]
    transformed_vertices = np.array(data[chosen_key])

    window = o3d.visualization.gui.Application.instance.create_window("SMPL Visualization", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(widget3d)

    mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(transformed_vertices), triangles=o3d.utility.Vector3iVector(faces))
    mesh.compute_vertex_normals()

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLitTransparency"
    material.base_color = [0.5, 0.5, 0.5, 0.8]
    widget3d.scene.add_geometry("mesh", mesh, material)

    bounds = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bounds, bounds.get_center())

    save_point_cloud(transformed_vertices, f"{chosen_key}.ply")

    o3d.visualization.gui.Application.instance.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and visualize mesh from JSON data.")
    parser.add_argument("--index", type=int, default=0, help="Index of the mesh to visualize.")
    args = parser.parse_args()
    main(args.index)
