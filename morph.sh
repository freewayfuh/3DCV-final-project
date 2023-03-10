PCD1=pcd/freeway/face.ply
PCD2=pcd/bwowen/face.ply

# Kmeans_1 method
python Kmeans.py --pcd1 $PCD1 --pcd2 $PCD2 --mapping --mapping_file result/Kmeans_1/Kmeans_1.pickle --kmean_level 1 --output_folder result/Kmeans_1/morph_ply

# Kmeans_3 method
python Kmeans.py --pcd1 $PCD1 --pcd2 $PCD2 --mapping --mapping_file result/Kmeans_3/Kmeans_3.pickle --kmean_level 3 --output_folder result/Kmeans_3/morph_ply

# axis method
python Kmeans.py --pcd1 $PCD1 --pcd2 $PCD2 --mapping --mapping_file result/axis/axis.pickle --output_folder result/axis/morph_ply

# distance method
python distance_method.py --pcd1 $PCD1 --pcd2 $PCD2 --mapping --mapping_file result/distance/distance.pickle --output_folder result/distance/morph_ply