import json
import numpy as np
import os
import torch
from tqdm import tqdm

def get_center_and_diag(cam_centers):
    cam_centers = np.hstack(cam_centers)
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    return center.flatten(), diagonal

def getNerfppNorm(input_filepath):
    with open(input_filepath, 'r') as json_file:
        data = json.load(json_file)

    cam_centers = []

    count = 0

    for pic_name in data.keys():
        count += 1

        R = np.array(data[pic_name]['R']) 
        T = np.array(data[pic_name]['T'])
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = T
        Rt[3, 3] = 1.0

        W2C = np.float32(Rt)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    print(translate)
    print(radius)
    print(count)


if __name__ == "__main__":
    input_filepath = 'inv_matrix.json'
    getNerfppNorm(input_filepath)
