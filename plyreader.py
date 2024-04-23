import open3d as o3d

def read_ply_and_analyze(file_path):
    point_cloud = o3d.io.read_point_cloud(file_path)
    
    print("Numbers of points:", len(point_cloud.points))

    if point_cloud.colors:
        print("Points have colors!")
        for i, color in enumerate(point_cloud.colors[:5]):
            print(f" {i} 's color: {color}")
    else:
        print("Points have no colors!")

    o3d.visualization.draw_geometries([point_cloud])

def main():
    file_path = 'lego\points3d.ply'
    read_ply_and_analyze(file_path)

if __name__ == "__main__":
    main()
