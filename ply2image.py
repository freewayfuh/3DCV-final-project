import open3d as o3d
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--ply_file", type=str, default='result/distance/morph_ply/0.ply')
parser.add_argument("--output", type=str, default='result/distance/morph_png/0.png')
parser.add_argument("--cam_param", type=str, default='cam.json')
args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.ply_file)
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
ctr = vis.get_view_control()
parameters = o3d.io.read_pinhole_camera_parameters(args.cam_param)
ctr.convert_from_pinhole_camera_parameters(parameters)
vis.capture_screen_image(args.output, do_render=True)
vis.destroy_window()