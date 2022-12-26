import open3d as o3d
import os
import pickle
import numpy as np

from tqdm.auto import tqdm
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--pcd1", type=str, required=True, default='pcd/freeway/face.ply')
parser.add_argument("--pcd2", type=str, required=True, default='pcd/bwowen/face.ply')
parser.add_argument("-k", type=int, default=60, help="The number of interpolation images to calculate")
parser.add_argument("--mapping", action='store_true', help="Recalculate the mapping")
parser.add_argument("--mapping_file", type=str, required=True, help="The mapping file")
parser.add_argument("--output_folder", type=str, required=True, help="The folder where .npy files are stored")
args = parser.parse_args()


def points_mapping(points1, points2):
    mapping = []

    # if len(points1) < len(points2):
    #     points1, points2 = points2, points1
    # assert len(points1) >= len(points2)
    # print(len(points1))
    # print(len(points2))
    # exit()

    is_mapped = set({})

    if len(points1) < len(points2):

        for i in tqdm(range(len(points1))):
            tmp = points2 - points1[i]
            tmp = tmp * tmp
            tmp = np.sum(tmp,axis=1)
            arg = np.argmin(tmp)
            mapping.append((i, arg))
            is_mapped.add(arg)


        for i in tqdm(range(len(points2))):
            if i in is_mapped:
                continue
            tmp = points1 - points2[i]
            tmp = tmp * tmp
            tmp = np.sum(tmp,axis=1)
            mapping.append((np.argmin(tmp), i))
    else:

        for i in tqdm(range(len(points2))):
            tmp = points1 - points2[i]
            tmp = tmp * tmp
            tmp = np.sum(tmp,axis=1)
            arg = np.argmin(tmp)
            mapping.append((arg, i))
            is_mapped.add(arg)

        for i in tqdm(range(len(points1))):
            if i in is_mapped:
                continue
            tmp = points2 - points1[i]
            tmp = tmp * tmp
            tmp = np.sum(tmp,axis=1)
            mapping.append((i, np.argmin(tmp)))
    
    return mapping


if __name__=='__main__':

    pcd1 = o3d.io.read_point_cloud(args.pcd1)
    pcd2 = o3d.io.read_point_cloud(args.pcd2)

    # (mean1, cov) = pcd1.compute_mean_and_covariance()
    # (mean2, cov) = pcd2.compute_mean_and_covariance()
    
    # mean_dif = mean1 - mean2
    # points = np.asarray(pcd1.points)
    # points = points - mean_dif
    # pcd1.points = o3d.utility.Vector3dVector(points)

    pcd1 = [pcd1]
    pcd2 = [pcd2]

    if args.mapping:

        print('==================================')
        print('Calculating points mapping...')
        print('==================================')

        total_mapping = []
        for i in range((len(pcd1))):
            p1 = np.asarray(pcd1[i].points)
            p2 = np.asarray(pcd2[i].points)
            temp_mapping = points_mapping(p1, p2)
            total_mapping.append(temp_mapping)
        
        print('==================================')
        print('Saving mapping file...')
        print('==================================')

        with open(args.mapping_file, 'wb') as f:
            pickle.dump(total_mapping, f)

    assert os.path.exists(args.mapping_file), 'Mapping file not found, please retype'
    with open(args.mapping_file, 'rb') as f:
        total_mapping = pickle.load(f)

    var_exists = 'total_mapping' in locals() or 'total_mapping' in globals()
    assert var_exists, 'total_mapping not exists, please use the argument "--mapping" to calculate the mapping'
    
    ## Interpolation!!!
    
    k = args.k - 1

    print('==================================')
    print('Calculating interpolation...')
    print(f'interpolating total {k+1} frames...')
    print('==================================')

    for r in range(0, k+1):
        points = []
        colors = []
        for i in range(len(pcd1)):
            p1 = np.asarray(pcd1[i].points)
            p2 = np.asarray(pcd2[i].points)
            c1 = np.asarray(pcd1[i].colors)
            c2 = np.asarray(pcd2[i].colors)
            ps = [(p1[m[0]]*(k-r) + p2[m[1]]*r)/k for m in total_mapping[i]]
            cs = [(c1[m[0]]*(k-r) + c2[m[1]]*r)/k for m in total_mapping[i]]
            if len(points) == 0:
                points = ps
                colors = cs
            else:
                points = np.concatenate((points, ps))
                colors = np.concatenate((colors, cs))
        
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(points)
        p.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(args.output_folder, f'{r}.ply'), p)