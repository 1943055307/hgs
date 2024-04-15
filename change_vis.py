import json
import open3d as o3d
import numpy as np
import torch
import pickle

def apply_transformation(vertices, joints, transform_matrix):
    vertices_torch = torch.tensor(vertices, dtype=torch.float32)
    joints_torch = torch.tensor(joints, dtype=torch.float32)
    transformed_vertices = torch.mm(transform_matrix[:3, :3], vertices_torch.T).T + transform_matrix[:3, 3]
    transformed_joints = torch.mm(transform_matrix[:3, :3], joints_torch.T).T + transform_matrix[:3, 3]
    return transformed_vertices.numpy(), transformed_joints.numpy()

def main():
    o3d.visualization.gui.Application.instance.initialize()

    model_path = 'smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    with open(model_path, 'rb') as f:
        smpl_model_data = pickle.load(f, encoding='latin1')
    faces = smpl_model_data['f']

    pic_name = 'r_002'

    json_file_path = 'transformed_pose_data.json'
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)[pic_name]

    transformed_vertices = data['transformed_vertices']
    transformed_joints = data['transformed_joints']

    inv_json_file_path = 'inv_matrix.json'
    with open(inv_json_file_path, 'r') as inv_json_file:
        inv_data = json.load(inv_json_file)[pic_name]

    inv_matrix = torch.tensor(inv_data['inv_matrix'], dtype=torch.float32)

    transformed_vertices, transformed_joints = apply_transformation(transformed_vertices, transformed_joints, inv_matrix)

    # print(transformed_joints)

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

    joints_material = o3d.visualization.rendering.MaterialRecord()
    joints_material.shader = "defaultLit"
    joints_material.base_color = [1, 0, 0, 0.6]

    for idx, joint in enumerate(transformed_joints):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(joint)
        sphere.paint_uniform_color([1, 0, 0])
        widget3d.scene.add_geometry(f"joint_sphere_{idx}", sphere, joints_material)

    bounds = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bounds, bounds.get_center())

    o3d.visualization.gui.Application.instance.run()

if __name__ == "__main__":
    main()
