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

    json_file_path1 = 'canonical_mesh_jump.json'
    json_file_path2 = 'canonical_mesh_jump_weighted.json'
    with open(json_file_path1, 'r') as json_file1, open(json_file_path2, 'r') as json_file2:
        data1 = json.load(json_file1)
        data2 = json.load(json_file2)

    window = o3d.visualization.gui.Application.instance.create_window("SMPL Visualization", 2048, 768)

    widget3d_1 = o3d.visualization.gui.SceneWidget()
    widget3d_1.scene = o3d.visualization.rendering.Open3DScene(window.renderer)

    widget3d_2 = o3d.visualization.gui.SceneWidget()
    widget3d_2.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    
    layout = o3d.visualization.gui.Horiz(0)
    layout.add_child(widget3d_1)
    layout.add_child(widget3d_2)
    window.add_child(layout)

    mesh_keys1 = list(data1.keys())
    mesh_keys2 = list(data2.keys())
    mesh_count = 0
    max_models = min(max_models, len(mesh_keys1), len(mesh_keys2))

    transition_start_time = time.time()

    flag = 0

    while True:
        o3d.visualization.gui.Application.instance.run_one_tick()
        current_time = time.time()
        alpha = (current_time - transition_start_time) / transition_duration
        if alpha > 1.0:
            alpha = 1.0
            transition_start_time = current_time
            current_mesh_key1 = mesh_keys1[mesh_count]
            next_mesh_key1 = mesh_keys1[(mesh_count + 1) % max_models]
            current_mesh_key2 = mesh_keys2[mesh_count]
            next_mesh_key2 = mesh_keys2[(mesh_count + 1) % max_models]
            mesh_count = (mesh_count + 1) % max_models
        else:
            current_mesh_key1 = mesh_keys1[mesh_count - 1] if mesh_count > 0 else mesh_keys1[-1]
            next_mesh_key1 = mesh_keys1[mesh_count]
            current_mesh_key2 = mesh_keys2[mesh_count - 1] if mesh_count > 0 else mesh_keys2[-1]
            next_mesh_key2 = mesh_keys2[mesh_count]

        if widget3d_1.scene.has_geometry("animated_mesh_1"):
            if alpha != 1.0:
                widget3d_1.scene.remove_geometry("animated_mesh_1")
        if widget3d_2.scene.has_geometry("animated_mesh_2"):
            if alpha != 1.0:
                widget3d_2.scene.remove_geometry("animated_mesh_2")

        vertices_start1 = np.array(data1[current_mesh_key1])
        vertices_end1 = np.array(data1[next_mesh_key1])
        interpolated_vertices1 = interpolate_mesh(vertices_start1, vertices_end1, alpha)
        mesh1 = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(interpolated_vertices1),
            triangles=o3d.utility.Vector3iVector(faces)
        )
        mesh1.compute_vertex_normals()

        vertices_start2 = np.array(data2[current_mesh_key2])
        vertices_end2 = np.array(data2[next_mesh_key2])
        interpolated_vertices2 = interpolate_mesh(vertices_start2, vertices_end2, alpha)
        mesh2 = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(interpolated_vertices2),
            triangles=o3d.utility.Vector3iVector(faces)
        )
        mesh2.compute_vertex_normals()

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = [0.5, 0.5, 0.5, 0.5]
        widget3d_1.scene.add_geometry("animated_mesh_1", mesh1, material)
        widget3d_2.scene.add_geometry("animated_mesh_2", mesh2, material)

        if flag == 0:
            center = widget3d_1.scene.bounding_box.get_center()
            eye = center + [0, -2, 0] 
            up = [0, 0, 1] 
            widget3d_1.setup_camera(60.0, widget3d_1.scene.bounding_box, center)
            widget3d_1.scene.camera.look_at(center, eye, up)
            widget3d_2.setup_camera(60.0, widget3d_2.scene.bounding_box, center)
            widget3d_2.scene.camera.look_at(center, eye, up)
            flag = 1

        camera_1 = widget3d_1.scene.camera
        camera_2 = widget3d_2.scene.camera
        camera_2.copy_from(camera_1)

        time.sleep(0.01)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL Mesh Visualization with Animated Transitions")
    parser.add_argument("-n", "--num", type=int, default=100, help="Maximum number of models to display")
    parser.add_argument("-t", "--transition_duration", type=float, default=1.0, help="Duration of transition between models in seconds")
    args = parser.parse_args()
    main(args.num, args.transition_duration)
