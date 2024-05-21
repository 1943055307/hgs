import json
import numpy as np
import open3d as o3d
import random

def visualize_joints(file_path, num=None, joint_names=None):
    with open(file_path, 'r') as f:
        data = json.load(f)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Joint Visualization")

    count = 0

    for pic_name, joints_data in data.items():
        if num is not None and count >= num:
            break

        transformation_matrices = np.array(joints_data['weighted_transformation_matrices'])

        joints = transformation_matrices[:, :3, 3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(joints)

        vis.add_geometry(pcd)

        color = [random.random(), random.random(), random.random()]

        for i, joint in enumerate(joints):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.paint_uniform_color(color)
            sphere.translate(joint)
            vis.add_geometry(sphere)
            if joint_names is not None:
                print(f"{joint_names[i]}: {joint}")

        count += 1

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    file_path = 'canonical_joints_jump_weighted.json'
    joint_names = [
        "Pelvis", "Left Hip", "Right Hip", "Spine1", "Left Knee", "Right Knee",
        "Spine2", "Left Ankle", "Right Ankle", "Spine3", "Left Foot", "Right Foot",
        "Neck", "Left Collar", "Right Collar", "Head", "Left Shoulder", "Right Shoulder",
        "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hand", "Right Hand"
    ]

    num = 4

    visualize_joints(file_path, num, joint_names)
