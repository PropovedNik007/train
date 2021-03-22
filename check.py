import cv2
import numpy as np
from infer_utils.video_writer import Video_holder

PATH_TO_VID = '/media/user3070/data/Teplovoz_video/sim_new2.mkv'

cap = cv2.VideoCapture(PATH_TO_VID)
vid = Video_holder('/media/user3070/data/Teplovoz_video/sim_new2HD.avi',1280, 720)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.resize(frame, (1280, 720),  interpolation=cv2.INTER_AREA)

        vid.add_frame(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
vid.save_video()
