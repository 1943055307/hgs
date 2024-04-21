import json
import open3d as o3d
import numpy as np
import torch
import pickle
import torchgeometry as tgm
from smplpytorch.pytorch.smpl_layer import SMPL_Layer

def apply_transformation(joints, transform_matrix):
    joints_torch = torch.tensor(joints, dtype=torch.float32)
    transformed_joints = torch.mm(transform_matrix[:3, :3], joints_torch.T).T + transform_matrix[:3, 3]
    return transformed_joints.numpy()

def main():
    o3d.visualization.gui.Application.instance.initialize()

    json_file_path = 'keypoints_data.json'
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)['r_001']

    transform_matrix = torch.tensor([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ])
    
    joint_coordinates = np.array(data['pred_xyz_jts_24_struct']).reshape(24, 3)
    transformed_joints = apply_transformation(joint_coordinates, transform_matrix)

    window = o3d.visualization.gui.Application.instance.create_window("SMPL Visualization", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(widget3d)

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
