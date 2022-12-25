import open3d as o3d
import json
import numpy as np

from scipy.spatial.transform import Rotation as R


K = 18
THETA = np.pi / 2
pcd = o3d.io.read_point_cloud('pcd/freeway/face.ply')

with open('camera/left.json', 'r') as jsonfile:
    left_param = json.load(jsonfile)

with open('camera/right.json', 'r') as jsonfile:
    right_param = json.load(jsonfile)

left_coor = left_param["extrinsic"][-4:-1]
right_coor = right_param["extrinsic"][-4:-1]
mean, cov = pcd.compute_mean_and_covariance()
# mean = np.array([0, 0, 0])

vector_1 = left_coor - mean
vector_2 = right_coor - mean
norm = np.cross(vector_1, vector_2)
norm = norm / np.linalg.norm(norm)

delta_theta = THETA / (K - 1)
for i, theta in enumerate([k * delta_theta for k in range(K)]):
    q = [*list(norm * np.sin(theta / 2)), np.cos(theta / 2)]
    r = R.from_quat(q).as_matrix()
    coor = r @ vector_1
    coor = coor + mean
    r = [*r[0], 0.0, *r[1], 0.0, *r[2], 0.0]
    with open(f'camera/move/{i}.json', 'w') as jsonfile:
        left_param["extrinsic"][:12] = r
        left_param["extrinsic"][-4:-1] = coor
        json.dump(left_param, jsonfile, indent=4)