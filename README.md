所有步驟：
1. 錄製完影片使用 VideoToFrame.py 把影片轉成 image sequance
2. 把轉出來的 image sequance 輸入至 COLMAP，製作 point cloud (輸出的檔案會在 COLMAP 專案裡的 dense/0/meshed-poisson.ply)
3. 使用 dbscan.py 做 outlier points removal (輸出的檔案會在 pcd/[user]/denoise.ply)
4. 使用 MeshLab 把人臉部分的 point cloud extract 出來
5. 使用 3D_transformation.py 做 point cloud alignment，方法包含 3D bounding box alignment 以及手動對齊 (輸出的檔案會在 pcd/[user]/face.ply)
6. 使用 readply.ipynb 調整 camera 的拍攝角度，選好適當的角度後輸出相機參數的 json 檔 (我們對好的檔案是 cam.json)
7. 使用 Kmeans.py 輸出 morphing 每一個 frame 的 .ply 檔案
8. 使用 ply2image.py 讀取 morphing 後的 .ply 檔案，使用相機參數拍照並存成 .png 檔案
9. 使用 image2video.py 把 image sequance 轉成 video

* 因為 image sequance 和 COLMAP 檔案極大，所以沒有上傳，我們直接提供原始的 video、原始的 .ply 檔以及前處理完的 .ply 檔


# Reproduce
```
bash Kmeans.sh
bash run.sh METHOD
```