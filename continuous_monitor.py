# code adapted from https://stackoverflow.com/questions/55411269/recording-video-in-specified-time-intervals-and-then-saving-them-into-file-openc

import numpy as np
import cv2
import time
import os
import random
import sys
import datetime


fps = 30
# width = 915
# height = 1626
video_codec = cv2.VideoWriter_fourcc(*'mp4v')

vid_len = 2.5

# name = random.randint(0, 1000)
# print(name)
# if os.path.isdir(str(name)) is False:
#     name = random.randint(0, 1000)
#     name = str(name)

name = "auto_videos"
print("All logs saved in dir:", name)
#os.mkdir(name)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,fps)
# ret = cap.set(3, width)
# ret = cap.set(4, height)
cur_dir = os.path.dirname(os.path.abspath(sys.argv[0]))


start = time.time()
video_file = os.path.join(name, str(datetime.datetime.now()) + ".mp4")
print("Capture video saved location : {}".format(video_file))

# Create a video write before entering the loop
video_writer = cv2.VideoWriter(
    video_file, video_codec, fps, (int(cap.get(3)), int(cap.get(4)))
)

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow("frame", frame)
        if time.time() - start > vid_len:
            start = time.time()
            video_file = os.path.join(name, str(datetime.datetime.now()) + ".mp4")
            video_writer = cv2.VideoWriter(
                video_file, video_codec, fps, (int(cap.get(3)), int(cap.get(4)))
            )
            # No sleeping! We don't want to sleep, we want to write
            # time.sleep(2.5)

        # Write the frame to the current video writer
        video_writer.write(frame)
        # if cv2.waitKey(1) == ord("q"):
        #     break
    else:
        break
cap.release()
cv2.destroyAllWindows()