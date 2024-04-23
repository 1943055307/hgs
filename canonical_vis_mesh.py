import json
import open3d as o3d
import numpy as np
import pickle
import argparse

def main(max_models):
    o3d.visualization.gui.Application.instance.initialize()

    model_path = 'smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
    with open(model_path, 'rb') as f:
        smpl_model_data = pickle.load(f, encoding='latin1')
    faces = smpl_model_data['f']

    json_file_path = 'canonical_mesh.json'
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    window = o3d.visualization.gui.Application.instance.create_window("SMPL Visualization", 1024, 768)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(widget3d)

    count = 0  
    for key, vertices in data.items():
        if count >= max_models:  
            break
        transformed_vertices = vertices
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(transformed_vertices),
            triangles=o3d.utility.Vector3iVector(faces)
        )
        mesh.compute_vertex_normals()

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = [0.5, 0.5, 0.5, 0.5]  # 半透明灰色

        widget3d.scene.add_geometry(f"mesh_{key}", mesh, material)
        count += 1  # 增加模型计数

    bounds = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bounds, bounds.get_center())

    o3d.visualization.gui.Application.instance.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL Mesh Visualization")
    parser.add_argument("-n", "--num", type=int, default=10, help="Maximum number of models to display")
    args = parser.parse_args()
    main(args.num)
