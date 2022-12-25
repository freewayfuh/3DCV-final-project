import cv2
import glob
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--png_folder", type=str, default='result/distance/morph_png')
parser.add_argument("--output", type=str, default='result/distance/demo.mp4')
parser.add_argument("--frame_rate", type=int, default=20)
args = parser.parse_args()


frameSize = (1920, 1080)
out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), args.frame_rate, frameSize)

frames = sorted(glob.glob(f'{args.png_folder}/*.png'), key = lambda x : int(x.split('/')[-1].split('.')[0]))

frames.insert(0, f'{args.png_folder}/0.png')
frames.insert(0, f'{args.png_folder}/0.png')
frames.insert(0, f'{args.png_folder}/0.png')
frames.insert(0, f'{args.png_folder}/0.png')
frames.insert(0, f'{args.png_folder}/0.png')
frames.append(f'{args.png_folder}/58.png')
frames.append(f'{args.png_folder}/58.png')
frames.append(f'{args.png_folder}/58.png')
frames.append(f'{args.png_folder}/58.png')
frames.append(f'{args.png_folder}/58.png')

frames_rev = sorted(glob.glob(f'{args.png_folder}/*.png'), key = lambda x : int(x.split('/')[-1].split('.')[0]), reverse=True)
frames_rev.remove(f'{args.png_folder}/59.png')
frames.extend(frames_rev)

for filename in frames:
    print(filename)
    img = cv2.imread(filename)
    out.write(img)

out.release()