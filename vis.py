import json
import open3d as o3d
import numpy as np
import torch
import pickle
import torchgeometry as tgm
from smplpytorch.pytorch.smpl_layer import SMPL_Layer

def main():
    o3d.visualization.gui.Application.instance.initialize()

    model_path = 'smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    with open(model_path, 'rb') as f:
        smpl_model_data = pickle.load(f, encoding='latin1')
    faces = smpl_model_data['f']

    json_file_path = 'transformed_pose_data.json'
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)['r_000']

    transformed_vertices = data['transformed_vertices']
    transformed_joints = data['transformed_joints']

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
