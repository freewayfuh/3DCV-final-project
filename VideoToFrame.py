import cv2

interval = 1
frame_count = 0
video_cap = cv2.VideoCapture('./video/bwowen.MOV')

while(True):
    ret, frame = video_cap.read()
    if ret is False:
        break

    cv2.imwrite(f'./images/bwowen/{frame_count}.png', cv2.flip(frame, 0))
    frame_count += 1