import json
import numpy as np
import open3d as o3d
import pickle
import argparse 
from plyfile import PlyData, PlyElement

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    
def save_point_cloud(vertices, filename):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)

    o3d.io.write_point_cloud(filename, point_cloud, write_ascii=True, print_progress=True)


def main(chosen_index):
    o3d.visualization.gui.Application.instance.initialize()

    model_path = 'smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    with open(model_path, 'rb') as f:
        smpl_model_data = pickle.load(f, encoding='latin1')
    faces = smpl_model_data['f']

    json_file_path = 'canonical_mesh.json'
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    keys = list(data.keys())
    if chosen_index < 0 or chosen_index >= len(keys):
        print(f"Invalid index {chosen_index}. Please choose an index between 0 and {len(keys) - 1}.")
        return

    chosen_key = keys[chosen_index]
    transformed_vertices = data[chosen_key]

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
