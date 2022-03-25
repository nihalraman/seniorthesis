# extract facemesh marker points as np array of dimension (# frames, # of landmarks, 2)
# input video path as command line argument

import cv2
import mediapipe as mp
import numpy as np
import sys, getopt, os
import pickle
import pandas as pd

path = f"/Users/nraman/Documents/GitHub/seniorthesis/auto_videos_temp/{sys.argv[1]}.mp4"
cap = cv2.VideoCapture(path)

scaler = pickle.load(open("scaler_model.sav", 'rb'))
rocket = pickle.load(open("minirocket_model.sav", 'rb'))
model = pickle.load(open("lda_model.sav", 'rb'))

num_landmarks = 468

# initialize FaceMesh modules
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
facemesh = mp_face_mesh.FaceMesh()
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)



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

landmarks = np.array(arrays)

# get relevant indices
lower = [76, 77, 90, 180, 85, 16, 315, 404, 320, 307]
upper = [184, 74, 73, 72, 11, 302, 303, 304, 408, 306]
u2 = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
l2 = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
u3 = [57, 186, 92, 165, 167, 164, 393, 391, 322, 410]
l3 = [287, 273, 335, 406, 313, 18, 83, 182, 106, 43]
combo_indices = lower + upper + u2 + l2 + u3 + l3

landmarks = landmarks[:, combo_indices]
landmarks = landmarks.reshape(landmarks.shape[0], len(combo_indices)*2)
maxlen = 76
const = 0
# make dataframe
df = pd.DataFrame(np.zeros([1, len(combo_indices)*2])).astype(object)
for a in range(len(combo_indices)*2):
  cur = landmarks[:, a]
  df.iloc[0, a] = np.pad(cur, (0, maxlen - len(cur)), constant_values = (const))



print(model.predict(scaler.transform(rocket.transform(df))))





