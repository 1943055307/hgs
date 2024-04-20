import json
import open3d as o3d
import numpy as np
import torch
import pickle
import torchgeometry as tgm
from smplpytorch.pytorch.smpl_layer import SMPL_Layer

def main():
    o3d.visualization.gui.Application.instance.initialize()

    json_file_path = 'canonical_joints.json'

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # joints = data['r_000']['transformed_joints'] + data['r_003']['transformed_joints'] 
    joints = data['r_000'] + data['r_001'] 

    window = o3d.visualization.gui.Application.instance.create_window("SMPL Visualization", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(widget3d)

    joints_material = o3d.visualization.rendering.MaterialRecord()
    joints_material.shader = "defaultLit"
    joints_material.base_color = [1, 0, 0, 0.6]

    for idx, joint in enumerate(joints):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(joint)
        sphere.paint_uniform_color([1, 0, 0])
        widget3d.scene.add_geometry(f"joint_sphere_{idx}", sphere, joints_material)

    bounds = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bounds, bounds.get_center())

    o3d.visualization.gui.Application.instance.run()

if __name__ == "__main__":
    main()
