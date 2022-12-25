import open3d as o3d
import numpy as np
import argparse
import copy

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def get_rotation_from_unit(v1, v2):
    v = np.cross(v1, v2)
    s = v/np.linalg.norm(v)
    c = np.dot(v1, v2)
    vx = np.array([
        [0, -1*v[2], v[1]],
        [v[2], 0, -1*v[0]],
        [-1*v[1], v[0], 0],
    ])
    I = np.array([[1,0,0],[0,1,0],[0,0,1]])
    R = I + vx + vx@vx*(1-c)/np.dot(s, s)
    return R

if __name__=='__main__':

    # python vis_ply.py -i face.ply -p pcd/freeway/
    # python vis_ply.py -i face.ply -p pcd/bwowen/

    # parser = argparse.ArgumentParser(description='run cluster')
    # parser.add_argument('-i', metavar = 'point cloud file', required = True, type = str,
    #                     help = 'point cloud file')
    # parser.add_argument('-p', metavar = 'target person folder', required = True, type = str,
    #                     help = 'target person folder')
    # args = parser.parse_args()


    pcd1 = o3d.io.read_point_cloud('pcd/bwowen/denoise.ply')
    pcd2 = o3d.io.read_point_cloud('pcd/freeway/denoise.ply')

    bbb = pcd1.get_oriented_bounding_box()
    fbb = pcd2.get_oriented_bounding_box()
    bbb.color = [1.0, 0.0, 0.0]
    fbb.color = [0.0, 1.0, 0.0]
    bbbp = np.asarray(bbb.get_box_points())
    fbbp = np.asarray(fbb.get_box_points())

    # pcd1a = o3d.geometry.PointCloud()
    # pcd1b = o3d.geometry.PointCloud()
    # pcd2a = o3d.geometry.PointCloud()
    # pcd2b = o3d.geometry.PointCloud()
    # pcd1a.points = o3d.utility.Vector3dVector(np.asarray(pcd1.points)[:40000])
    # pcd1a.colors = o3d.utility.Vector3dVector(np.asarray(pcd1.colors)[:40000])
    # pcd1b.points = o3d.utility.Vector3dVector(np.asarray(pcd1.points)[40000:])
    # pcd1b.colors = o3d.utility.Vector3dVector(np.asarray(pcd1.colors)[40000:])
    # pcd2a.points = o3d.utility.Vector3dVector(np.asarray(pcd2.points)[:200000])
    # pcd2a.colors = o3d.utility.Vector3dVector(np.asarray(pcd2.colors)[:200000])
    # pcd2b.points = o3d.utility.Vector3dVector(np.asarray(pcd2.points)[200000:])
    # pcd2b.colors = o3d.utility.Vector3dVector(np.asarray(pcd2.colors)[200000:])

    # o3d.visualization.draw_geometries([pcd1a, pcd1b, pcd2a, pcd2b, bbb, fbb])

    # print(bbbp)
    # print(fbbp)
    
    ### Check Bounding Box Points index
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        # bbbp[6],
        # fbbp[6],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        # [0, 4],
        # [0, 5],
    ]
    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        # [1, 0, 0],
        # [0, 1, 0],
    ]
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd1, pcd2, bbb, fbb, line_set])

    # bbbp = np.array([bbbp[7], bbbp[2], bbbp[1], bbbp[4], bbbp[3], bbbp[6], bbbp[5], bbbp[0]])
    bbbp = np.array([bbbp[6], bbbp[3], bbbp[4], bbbp[1], bbbp[2], bbbp[7], bbbp[0], bbbp[5]])
    R, t = rigid_transform_3D(bbbp.T, fbbp.T)
    pcd1.rotate(R)
    pcd1.translate(t)
    # pcd1.translate(1.65*(fbbp[3]-fbbp[0]))
    # pcd1.translate(0.1*(fbbp[3]-fbbp[5]))
    pcd1.translate(0.84*(fbbp[2]-fbbp[0]))
    pcd1.translate(0.75*(fbbp[0]-fbbp[1]))
    pcd1.translate(0.1*(fbbp[3]-fbbp[0]))

    ### Fit Rotation
    v1 = fbbp[2]-fbbp[0]
    v1 = v1/np.linalg.norm(v1)
    v2 = 0.25*(fbbp[7]-fbbp[2]) + (fbbp[2]-fbbp[0])
    v2 = v2/np.linalg.norm(v2)
    R = get_rotation_from_unit(v1, v2)
    pcd1.rotate(R)

    v1 = fbbp[2]-fbbp[0]
    v1 = v1/np.linalg.norm(v1)
    v2 = 0.09*(fbbp[5]-fbbp[2]) + (fbbp[2]-fbbp[0])
    v2 = v2/np.linalg.norm(v2)
    R = get_rotation_from_unit(v1, v2)
    pcd1.rotate(R)

    v1 = fbbp[3]-fbbp[0]
    v1 = v1/np.linalg.norm(v1)
    v2 = 0.06*(fbbp[6]-fbbp[3]) + (fbbp[3]-fbbp[0])
    v2 = v2/np.linalg.norm(v2)
    R = get_rotation_from_unit(v1, v2)
    pcd1.rotate(R)

    ### Move to origin
    (mean, cov) = pcd1.compute_mean_and_covariance()
    print(mean)
    points = np.asarray(pcd1.points)
    pcd1.points = o3d.utility.Vector3dVector(points-mean)
    points = np.asarray(pcd2.points)
    pcd2.points = o3d.utility.Vector3dVector(points-mean)

    (mean, cov) = pcd1.compute_mean_and_covariance()
    (mean2, cov2) = pcd2.compute_mean_and_covariance()
    print(mean, mean2)

    bbb = pcd1.get_oriented_bounding_box()
    bbb.color = [1.0, 0.0, 0.0]


    source_temp = copy.deepcopy(pcd1)
    target_temp = copy.deepcopy(pcd2)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    pcd1a = o3d.geometry.PointCloud()
    pcd1b = o3d.geometry.PointCloud()
    pcd2a = o3d.geometry.PointCloud()
    pcd2b = o3d.geometry.PointCloud()
    pcd1a.points = o3d.utility.Vector3dVector(np.asarray(source_temp.points)[:40000])
    pcd1a.colors = o3d.utility.Vector3dVector(np.asarray(source_temp.colors)[:40000])
    pcd1b.points = o3d.utility.Vector3dVector(np.asarray(source_temp.points)[40000:])
    pcd1b.colors = o3d.utility.Vector3dVector(np.asarray(source_temp.colors)[40000:])
    pcd2a.points = o3d.utility.Vector3dVector(np.asarray(target_temp.points)[:200000])
    pcd2a.colors = o3d.utility.Vector3dVector(np.asarray(target_temp.colors)[:200000])
    pcd2b.points = o3d.utility.Vector3dVector(np.asarray(target_temp.points)[200000:])
    pcd2b.colors = o3d.utility.Vector3dVector(np.asarray(target_temp.colors)[200000:])


    # o3d.visualization.draw_geometries([pcd1, pcd2, bbb, fbb])
    o3d.visualization.draw_geometries([pcd1a, pcd1b, pcd2a, pcd2b, bbb, fbb, line_set])

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd1)
    # vis.add_geometry(pcd2)
    # vis.add_geometry(bbb)
    # vis.add_geometry(fbb)
    # vis.add_geometry(line_set)
    # opt = vis.get_render_option()
    # opt.show_coordinate_frame = True
    # parameters = o3d.io.read_pinhole_camera_parameters('ScreenCamera_2022-12-17-22-10-06.json')
    # ctr = vis.get_view_control()
    # ctr.convert_from_pinhole_camera_parameters(parameters)
    # vis.capture_screen_image('morph/0.png', do_render=True)
    # vis.run()
    # vis.destroy_window()

    o3d.io.write_point_cloud('pcd/bwowen/face.ply', pcd1)
    o3d.io.write_point_cloud('pcd/freeway/face.ply', pcd2)