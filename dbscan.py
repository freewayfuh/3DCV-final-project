import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
from skimage import io

def run_clustering(pcd_path, label_read_path, label_write_path, class_we_want):

    ### Read Point Cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    org_colors = np.asarray(pcd.colors).copy() # later use 
    org_points = np.asarray(pcd.points).copy() # later use
    print(f"point cloud contains: {len(pcd.points)} points")
    # print(f"The center: {pcd.get_center()}")

    ### Clustering (or load label)
    if os.path.exists(label_read_path):
        print(f"find {label_read_path}, reading the pickle file...")
        with open(label_read_path, 'rb') as f:
            labels = pickle.load(f)
    else:
        print(f"{label_read_path} not exist, making new pickle file...")
        labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=50, print_progress=True))
        with open(label_read_path, 'wb') as f:
            pickle.dump(labels, f)
    
    max_label = labels.max()
    print(f"\npoint cloud contains {max_label + 1} clusters")
    hist, bin = np.histogram(labels, bins = np.arange(-1, max_label+1))
    print("cluster histogram: ", hist)

    ### Show class by colors
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    ### Visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    vis.run()
    vis.destroy_window()

    ### Show class & color mapping
    label_set = list(set(labels))
    color_set = plt.get_cmap("tab20")(label_set / (max_label if max_label > 0 else 1))
    color_set[np.array(label_set) < 0] = 0
    color_set = color_set[:, :3]
    print(color_set)
    idx = np.array([np.arange(max_label+2)])
    io.imshow(color_set[idx])
    plt.show()

    ### Choose the class we want
    # while True:
    #     class_we_want = int(input("Enter Class: "))
    #     if class_we_want == 1000:
    #         class_we_want = np.argmax(hist) - 1
    #         print('class_we_want: ', class_we_want)
    #     if class_we_want <= max_label and class_we_want >= -1: break

    org_colors = org_colors[labels == class_we_want]
    org_points = org_points[labels == class_we_want]
    pcd.colors = o3d.utility.Vector3dVector(org_colors)
    pcd.points = o3d.utility.Vector3dVector(org_points)

    if label_write_path == None:
        label_write_path = 'face.ply'
    
    ### Store it
    o3d.io.write_point_cloud(filename=label_write_path, pointcloud=pcd)
    ### Visualization again
    ### Segmentation fault 11 ??? Why ???


if __name__=='__main__':

    # eps = 0.05 min = 50
    # python dbscan.py -i pcd/freeway/meshed-poisson.ply -o pcd/freeway/denoise.ply -l pcd/freeway/label.pickle -c 3
    # python dbscan.py -i pcd/bwowen/meshed-poisson.ply -o pcd/bwowen/denoise.ply -l pcd/bwowen/label.pickle -c 206

    parser = argparse.ArgumentParser(description='run cluster')
    parser.add_argument('-i', metavar = 'read point cloud', required = True, type = str,
                        help = 'read point cloud')
    parser.add_argument('-o', metavar = 'write point cloud', required = True, type = str,
                        help = 'write point cloud')
    parser.add_argument('-l', metavar = 'read point cloud label', required = True, type = str,
                        help = 'read point cloud label')
    parser.add_argument('-c', metavar = 'point cloud class we want', required = True, type = int,
                        help = 'point cloud class we want')
    
    args = parser.parse_args()

    run_clustering(args.i, args.l, args.o, args.c)