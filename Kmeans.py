import open3d as o3d
import os
import pickle
import numpy as np

from sklearn.cluster import KMeans
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--pcd1", type=str, required=True, default='pcd/freeway/face.ply')
parser.add_argument("--pcd2", type=str, required=True, default='pcd/bwowen/face.ply')
parser.add_argument("--kmean_level", type=int, help="Recalculate the mapping")
parser.add_argument("-k", type=int, default=60, help="The number of interpolation images to calculate")
parser.add_argument("--mapping", action='store_true', help="Recalculate the mapping")
parser.add_argument("--mapping_file", type=str, required=True, help="The mapping file")
parser.add_argument("--output_folder", type=str, required=True, help="The folder where .npy files are stored")
args = parser.parse_args()


def pcd_kmeans(pcd, n_clusters=20):
    points = np.asarray(pcd.points)
    # print(f"point cloud contains: {len(points)} points")
    # print("calculateing kmeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(points)
    return kmeans

#################################
### hungarian_algorithm start ###
#################################

def min_zero_row(zero_mat, mark_zero):
	
	'''
	The function can be splitted into two steps:
	#1 The function is used to find the row which containing the fewest 0.
	#2 Select the zero number on the row, and then marked the element corresponding row and column as False
	'''

	#Find the row
	min_row = [99999, -1]

	for row_num in range(zero_mat.shape[0]): 
		if np.sum(zero_mat[row_num] == True) > 0 and min_row[0] > np.sum(zero_mat[row_num] == True):
			min_row = [np.sum(zero_mat[row_num] == True), row_num]

	# Marked the specific row and column as False
	zero_index = np.where(zero_mat[min_row[1]] == True)[0][0]
	mark_zero.append((min_row[1], zero_index))
	zero_mat[min_row[1], :] = False
	zero_mat[:, zero_index] = False

def mark_matrix(mat):

	'''
	Finding the returning possible solutions for LAP problem.
	'''

	#Transform the matrix to boolean matrix(0 = True, others = False)
	cur_mat = mat
	zero_bool_mat = (cur_mat == 0)
	zero_bool_mat_copy = zero_bool_mat.copy()

	#Recording possible answer positions by marked_zero
	marked_zero = []
	while (True in zero_bool_mat_copy):
		min_zero_row(zero_bool_mat_copy, marked_zero)
	
	#Recording the row and column positions seperately.
	marked_zero_row = []
	marked_zero_col = []
	for i in range(len(marked_zero)):
		marked_zero_row.append(marked_zero[i][0])
		marked_zero_col.append(marked_zero[i][1])

	#Step 2-2-1
	non_marked_row = list(set(range(cur_mat.shape[0])) - set(marked_zero_row))
	
	marked_cols = []
	check_switch = True
	while check_switch:
		check_switch = False
		for i in range(len(non_marked_row)):
			row_array = zero_bool_mat[non_marked_row[i], :]
			for j in range(row_array.shape[0]):
				#Step 2-2-2
				if row_array[j] == True and j not in marked_cols:
					#Step 2-2-3
					marked_cols.append(j)
					check_switch = True

		for row_num, col_num in marked_zero:
			#Step 2-2-4
			if row_num not in non_marked_row and col_num in marked_cols:
				#Step 2-2-5
				non_marked_row.append(row_num)
				check_switch = True
	#Step 2-2-6
	marked_rows = list(set(range(mat.shape[0])) - set(non_marked_row))

	return(marked_zero, marked_rows, marked_cols)

def adjust_matrix(mat, cover_rows, cover_cols):
	cur_mat = mat
	non_zero_element = []

	#Step 4-1
	for row in range(len(cur_mat)):
		if row not in cover_rows:
			for i in range(len(cur_mat[row])):
				if i not in cover_cols:
					non_zero_element.append(cur_mat[row][i])
	min_num = min(non_zero_element)

	#Step 4-2
	for row in range(len(cur_mat)):
		if row not in cover_rows:
			for i in range(len(cur_mat[row])):
				if i not in cover_cols:
					cur_mat[row, i] = cur_mat[row, i] - min_num
	#Step 4-3
	for row in range(len(cover_rows)):  
		for col in range(len(cover_cols)):
			cur_mat[cover_rows[row], cover_cols[col]] = cur_mat[cover_rows[row], cover_cols[col]] + min_num
	return cur_mat

def hungarian_algorithm(mat): 
	dim = mat.shape[0]
	cur_mat = mat

	#Step 1 - Every column and every row subtract its internal minimum
	for row_num in range(mat.shape[0]): 
		cur_mat[row_num] = cur_mat[row_num] - np.min(cur_mat[row_num])
	
	for col_num in range(mat.shape[1]): 
		cur_mat[:,col_num] = cur_mat[:,col_num] - np.min(cur_mat[:,col_num])
	zero_count = 0
	while zero_count < dim:
		#Step 2 & 3
		ans_pos, marked_rows, marked_cols = mark_matrix(cur_mat)
		zero_count = len(marked_rows) + len(marked_cols)

		if zero_count < dim:
			cur_mat = adjust_matrix(cur_mat, marked_rows, marked_cols)

	return ans_pos

def ans_calculation(mat, pos):
	total = 0
	ans_mat = np.zeros((mat.shape[0], mat.shape[1]))
	for i in range(len(pos)):
		total += mat[pos[i][0], pos[i][1]]
		ans_mat[pos[i][0], pos[i][1]] = mat[pos[i][0], pos[i][1]]
	return total, ans_mat

#################################
###  hungarian_algorithm end  ###
#################################

def map_centerd_hungarian(center1, center2):

    # print("RUNNING: Hungarian Algorithm / TOTAL MINIMUM DISTANCE")

    if len(center1) != len(center2):
        print("ERROR: center number not aligned!!!")

    dis = []
    for cen in center1:
        dis.append([np.linalg.norm(cen-cen2) for cen2 in center2])
    dis = np.array(dis)

    ans_pos = hungarian_algorithm(dis.copy())

    return ans_pos

def map_centerd(center1, center2):

    # print("RUNNING: SMALLEST DISTANCE FIRST")

    if len(center1) != len(center2):
        print("ERROR: center number not aligned!!!")

    dis = []
    for cen in center1:
        dis.append([np.linalg.norm(cen-cen2) for cen2 in center2])
    dis = np.array(dis)
    
    mapping = []
    while len(dis) > 0:
        temp = np.array([(min(d), np.argmin(d)) for d in dis])
        min_idx = np.argmin(temp[:,0])
        mapping.append((min_idx, temp[min_idx,1]))
        dis = np.delete(dis, min_idx, axis=0)
        dis = np.delete(dis, int(temp[min_idx,1]), axis=1)

    mapping = np.array(mapping)

    for i in range(len(mapping)):
        if i != 0:
            a = sorted(mapping[:i, 0])
            for j in range(len(a)):
                if mapping[i][0] >= a[j]:
                    mapping[i][0] = mapping[i][0] + 1
                else:
                    break

    for i in range(len(mapping)):
        if i != 0:
            a = sorted(mapping[:i, 1])
            for j in range(len(a)):
                if mapping[i][1] >= a[j]:
                    mapping[i][1] = mapping[i][1] + 1
                else:
                    break

    # print(mapping)
    return mapping

def points_mapping(points1, points2):
    ### map points1 to points2
    ### Axis Base Mapping
    mapping = []
    ratio = len(points2)/len(points1)
    last = 0
    current = 1
    for i in range(1,len(points1)+1):
        current = int(np.floor(i*ratio+0.0000001))
        for j in range(last+1, current+1):
            mapping.append((i-1, j-1))
        if current == 0:
            mapping.append((i-1, 0))
        elif last == current:
            mapping.append((i-1, last-1))
        last = current

    if len(points2) > len(points1):
        if len(mapping) == len(points2)-1:
            mapping.append(len(points1)-1, len(points2)-1)
        elif len(points2) > len(mapping):
            print('ERROR: points mapping ERROR')
            print(len(points2), len(mapping))
    else:
        if len(mapping) == len(points1)-1:
            mapping.append(len(points1)-1, len(points2)-1)
        elif len(points1) > len(mapping):
            print('ERROR: points mapping ERROR')
            print(len(points1), len(mapping))
    
    return mapping

def points_mapping_random(points1, points2):
    mapping = []
    ratio = len(points2)/len(points1)
    last = 0
    current = 1
    for i in range(1,len(points1)+1):
        current = int(np.floor(i*ratio+0.0000001))
        for j in range(last+1, current+1):
            mapping.append((i-1, j-1))
        if current == 0:
            mapping.append((i-1, 0))
        elif last == current:
            mapping.append((i-1, last-1))
        last = current

    if len(points2) > len(points1):
        if len(mapping) == len(points2)-1:
            mapping.append(len(points1)-1, len(points2)-1)
        elif len(points2) > len(mapping):
            print('ERROR: points mapping ERROR')
            print(len(points2), len(mapping))
    else:
        if len(mapping) == len(points1)-1:
            mapping.append(len(points1)-1, len(points2)-1)
        elif len(points1) > len(mapping):
            print('ERROR: points mapping ERROR')
            print(len(points1), len(mapping))
    
    return mapping

def do_kmeans_cluster_mapping(pcd1, pcd2):

    kmeans1 = pcd_kmeans(pcd1)
    labels1 = kmeans1.labels_
    centers1 = kmeans1.cluster_centers_
    # colors1 = plt.get_cmap("tab20")(labels1 / 20)
    # pcd1.colors = o3d.utility.Vector3dVector(colors1[:, :3])

    kmeans2 = pcd_kmeans(pcd2)
    labels2 = kmeans2.labels_
    centers2 = kmeans2.cluster_centers_
    # colors2 = plt.get_cmap("tab20")(labels2 / 20)
    # pcd2.colors = o3d.utility.Vector3dVector(colors2[:, :3])

    # mapping = map_centerd(centers, centers2)
    mapping = map_centerd_hungarian(centers1, centers2)

    temp_pcd1s = []
    temp_pcd2s = []
    for m in mapping:
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(np.asarray(pcd1.points)[labels1 == m[0]])
        p.colors = o3d.utility.Vector3dVector(np.asarray(pcd1.colors)[labels1 == m[0]])
        temp_pcd1s.append(p)
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(np.asarray(pcd2.points)[labels2 == m[1]])
        p.colors = o3d.utility.Vector3dVector(np.asarray(pcd2.colors)[labels2 == m[1]])
        temp_pcd2s.append(p)
    
    if len(temp_pcd1s) != len(temp_pcd2s):
        print("ERROR: PCD number not aligned!!!")

    return temp_pcd1s, temp_pcd2s



if __name__=='__main__':
    
    pcd1 = o3d.io.read_point_cloud(args.pcd1)
    pcd2 = o3d.io.read_point_cloud(args.pcd2)

    # (mean, cov) = pcd.compute_mean_and_covariance()
    # (mean2, cov) = pcd2.compute_mean_and_covariance()
    
    # mean_dif = mean - mean2
    # points = np.asarray(pcd.points)
    # points = points - mean_dif
    # pcd.points = o3d.utility.Vector3dVector(points)

    pcd1 = [pcd1]
    pcd2 = [pcd2]

    if args.mapping:

        kmean_level = 0 if args.kmean_level is None else args.kmean_level

        for kl in range(kmean_level):

            print('==================================')
            print(f'Calculating level {kl} kmeans...')
            print('==================================')

            temp_pcd1 = []
            temp_pcd2 = []
            for i in range(len(pcd1)):
                pcd1s, pcd2s = do_kmeans_cluster_mapping(pcd1[i], pcd2[i])
                temp_pcd1 = np.concatenate((temp_pcd1, pcd1s))
                temp_pcd2 = np.concatenate((temp_pcd2, pcd2s))

            pcd1 = temp_pcd1
            pcd2 = temp_pcd2

        print('==================================')
        print('Calculating points mapping...')
        print('==================================')

        total_mapping = []
        for i in range(len(pcd1)):
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


    ### Interpolation!!!

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