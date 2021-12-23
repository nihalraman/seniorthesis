# extract facemesh marker points as np array of dimension (# frames, # of landmarks, 2)
# input video path as command line argument

import cv2
import mediapipe as mp
import numpy as np
import sys, getopt, os

path = sys.argv[1]

num_landmarks = 478

# initialize FaceMesh modules
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
facemesh = mp_face_mesh.FaceMesh(refine_landmarks = True)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(path)

arrays = []

while True:
  # extract frames from single video
  ret, image = cap.read()
  # break if frame doesn't exist
  if ret is not True:
    break
  
  height, width, _ = image.shape

  # initialize array to hold landmarks (x, y)
  landmark_array = np.zeros((num_landmarks, 2))
  
  result = facemesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  
  for facial_landmarks in result.multi_face_landmarks:
    for i in range(num_landmarks):
        point = facial_landmarks.landmark[i]
        x = int(point.x * width)
        y = int(point.y * height)

        landmark_array[i, 0] = x
        landmark_array[i, 1] = y
        
  # add landmarks to array, show image with landmarks on it
  arrays.append(landmark_array)

new_filename = os.path.splitext(os.path.basename(path))[0]
new_path = f"/Users/nraman/Documents/thesis_videos/{new_filename}/{new_filename}_keypoints"

np.save(new_path, np.array(arrays))

