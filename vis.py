import json
import open3d as o3d
import numpy as np
import torch
import pickle
import torchgeometry as tgm
from smplpytorch.pytorch.smpl_layer import SMPL_Layer

def rotation_matrix2axis_angle(R):
    theta = torch.acos(torch.clamp((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2, -1.0, 1.0))
    axis = torch.stack([(R[:, 2, 1] - R[:, 1, 2]),
                        (R[:, 0, 2] - R[:, 2, 0]),
                        (R[:, 1, 0] - R[:, 0, 1])], dim=1) # rot axis
    sin_theta = torch.sin(theta)
    axis = torch.where(sin_theta.unsqueeze(-1) >= 1e-7, axis / (2 * sin_theta).unsqueeze(-1), axis)
    axis_angle = axis * theta.unsqueeze(-1)
    return axis_angle

def main():
    o3d.visualization.gui.Application.instance.initialize()

    model_path = 'smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    with open(model_path, 'rb') as f:
        smpl_model_data = pickle.load(f, encoding='latin1')
    faces = smpl_model_data['f']

    smpl_layer = SMPL_Layer(
        gender='female',
        model_root='smpl/models'
    )

    json_file_path = 'HybrIK/SMPL/keypoints_data.json'

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)['r_011'] 

    theta_params = torch.tensor(data['pred_theta_mats'], dtype=torch.float32)

    rotation_matrices = theta_params.view(24, 3, 3)

    axis_angles = torch.zeros(24, 3)

    for i in range(24):
        axis_angle = rotation_matrix2axis_angle(rotation_matrices[i].unsqueeze(0))
        axis_angles[i] = axis_angle.squeeze(0)

    pose_params = axis_angles.view(1, 72)

    shape_params = torch.tensor(data['pred_shape'], dtype=torch.float32)

    vertices, joints = smpl_layer(pose_params, shape_params)
    
    vertices = vertices.detach().numpy().squeeze()
    joints = joints.detach().numpy().squeeze()

    window = o3d.visualization.gui.Application.instance.create_window("SMPL Visualization", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(widget3d)

    mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices), triangles=o3d.utility.Vector3iVector(faces))
    mesh.compute_vertex_normals()

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLitTransparency"  
    material.base_color = [0.5, 0.5, 0.5, 0.8] 

    widget3d.scene.add_geometry("mesh", mesh, material)

    joints_material = o3d.visualization.rendering.MaterialRecord()
    joints_material.shader = "defaultLit" 
    joints_material.base_color = [1, 0, 0, 0.6] 

    for idx, joint in enumerate(joints):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(joint) 
        sphere.paint_uniform_color([1, 0, 0])
        widget3d.scene.add_geometry(f"joint_sphere_{idx}", sphere, joints_material)

    xpoints = torch.tensor(data['pred_xyz_jts_24_struct'], dtype=torch.float32).view(24, 3)

    xpoints_material = o3d.visualization.rendering.MaterialRecord()
    xpoints_material.shader = "defaultLit" 
    xpoints_material.base_color = [0, 0, 1, 0.6] 

    for idx, xpoint in enumerate(xpoints):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(xpoint) 
        sphere.paint_uniform_color([1, 0, 0])
        # widget3d.scene.add_geometry(f"xpoint_sphere_{idx}", sphere, xpoints_material)

    bounds = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bounds, bounds.get_center())

    o3d.visualization.gui.Application.instance.run()

if __name__ == "__main__":
    main()
