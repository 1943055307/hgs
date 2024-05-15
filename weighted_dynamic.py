import json
import open3d as o3d
import numpy as np
import pickle
import argparse
import time

def interpolate_mesh(vertices_start, vertices_end, alpha):
    return vertices_start * (1 - alpha) + vertices_end * alpha

def main(max_models, transition_duration=2.0):
    o3d.visualization.gui.Application.instance.initialize()

    model_path = 'smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    with open(model_path, 'rb') as f:
        smpl_model_data = pickle.load(f, encoding='latin1')
    faces = smpl_model_data['f']

    json_file_path = 'canonical_mesh_jump_weighted.json'
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    window = o3d.visualization.gui.Application.instance.create_window("SMPL Visualization", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(widget3d)

    mesh_keys = list(data.keys())
    mesh_count = 0
    max_models = min(max_models, len(mesh_keys))

    transition_start_time = time.time()

    flag = 0

    while True:
        o3d.visualization.gui.Application.instance.run_one_tick()
        current_time = time.time()
        alpha = (current_time - transition_start_time) / transition_duration
        if alpha > 1.0:
            alpha = 1.0
            transition_start_time = current_time
            current_mesh_key = mesh_keys[mesh_count]
            next_mesh_key = mesh_keys[(mesh_count + 1) % max_models]
            mesh_count = (mesh_count + 1) % max_models
        else:
            current_mesh_key = mesh_keys[mesh_count - 1] if mesh_count > 0 else mesh_keys[-1]
            next_mesh_key = mesh_keys[mesh_count]

        if widget3d.scene.has_geometry("animated_mesh"):
            if alpha != 1.0:
                widget3d.scene.remove_geometry("animated_mesh")

        vertices_start = np.array(data[current_mesh_key])
        vertices_end = np.array(data[next_mesh_key])
        interpolated_vertices = interpolate_mesh(vertices_start, vertices_end, alpha)
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(interpolated_vertices),
            triangles=o3d.utility.Vector3iVector(faces)
        )
        mesh.compute_vertex_normals()

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = [0.5, 0.5, 0.5, 0.5]
        widget3d.scene.add_geometry("animated_mesh", mesh, material)

        if flag == 0:
            bounds = widget3d.scene.bounding_box
            widget3d.setup_camera(60.0, bounds, bounds.get_center())
            flag = 1

        time.sleep(0.01) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL Mesh Visualization with Animated Transitions")
    parser.add_argument("-n", "--num", type=int, default=100, help="Maximum number of models to display")
    parser.add_argument("-t", "--transition_duration", type=float, default=1.0, help="Duration of transition between models in seconds")
    args = parser.parse_args()
    main(args.num, args.transition_duration)
