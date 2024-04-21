import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def main():
    o3d.visualization.gui.Application.instance.initialize()

    json_file_path = 'canonical_coor_joints.json'
    ignore_keys_file = 'ignore_keys.txt'

    with open(ignore_keys_file, 'r') as file:
        ignore_keys = {line.strip() for line in file} 

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    window = o3d.visualization.gui.Application.instance.create_window("SMPL Visualization", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(widget3d)

    cmap = plt.get_cmap("viridis")
    num_sets = 10

    for i in range(num_sets):
        key = f'r_{i:03d}'
        if key in ignore_keys: 
            continue
        if key in data:
            joints = np.array(data[key])
            color = cmap(i / num_sets)[:3]
            joints_material = o3d.visualization.rendering.MaterialRecord()
            joints_material.shader = "defaultLit"
            joints_material.base_color = [*color, 0.6]

            for idx, joint in enumerate(joints):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                sphere.translate(joint)
                sphere.paint_uniform_color(color)
                widget3d.scene.add_geometry(f"joint_sphere_{key}_{idx}", sphere, joints_material)

    bounds = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bounds, bounds.get_center())

    o3d.visualization.gui.Application.instance.run()

if __name__ == "__main__":
    main()
