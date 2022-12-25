import cv2
import glob
from PIL import Image, ImageFilter


def merge_images(image1, image2):
    """Merge two images into one vertical image
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    (width1, height1) = image1.size
    (width2, height2) = image2.size
    result_width = width1 + width2
    result_height = max(height1, height2)
    print (height2)
    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result

frameSize = (1920, 1080)
# out = cv2.VideoWriter('morphing/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, frameSize)

imgs = None
for i, filename in enumerate(sorted(glob.glob('morphing/images/*.png'), key = lambda x : int(x.split('/')[-1].split('.')[0]))):
    if (i % 3 == 0):
        print(filename)
        img = Image.open(filename)
        if imgs is not None:
            imgs = merge_images(imgs, img)
        else:
            imgs = img

imgs.save('demo.jpg', "JPEG")



# out.release()