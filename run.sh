# METHOD=Kmeans_1
# METHOD=Kmeans_3
# METHOD=axis
# METHOD=distance

METHOD=$1

for INDEX in {0..59}
do
    echo python ply2image.py --pcd result/$METHOD/morph_ply/$INDEX.ply --output result/$METHOD/morph_png/$INDEX.png --cam_param cam.json
    python ply2image.py --pcd $METHOD/morph_ply/$INDEX.ply --output $METHOD/morph_png/$INDEX.png --cam_param cam.json
done

echo python image2video.py --output result/$METHOD/demo.mp4
python image2video.py --output $METHOD/demo.mp4