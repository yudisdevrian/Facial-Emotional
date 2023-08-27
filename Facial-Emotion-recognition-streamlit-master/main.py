from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import copy
from deepface import DeepFace
from deepface.detectors import FaceDetector
import pandas as pd
import itertools
import mediapipe as mp
import numpy as np
from model import KeyPointClassifier

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_info_text(image, brect, facial_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    if facial_text != "":
        info_text = 'Emotion :' + facial_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image



# Model load
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) 

keypoint_classifier = KeyPointClassifier()

import csv

# Read labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

#keypoint_classifier = KeyPointClassifier()


class VideoProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")
		image = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		debug_image=copy.deepcopy(image)
		detector_name="opencv"
		detector = FaceDetector.build_model(detector_name) #set opencv, ssd, dlib, mtcnn or retinaface
		obj2 = FaceDetector.detect_faces(detector, detector_name, debug_image)
  
		if (len(obj2)>=1):
				print(len(obj2))
				for i in range(len(obj2)):
					cur=obj2[i][1]
					x=cur[0]
					y=cur[1]
					w=cur[2]
					h=cur[3]
					detected_face=debug_image[y:y+h,x:x+w]
					image_height, image_width, c = detected_face.shape
					cv2.rectangle(debug_image,(x,y),(x+w,y+h),(0,0,0),1)							                
					results = face_mesh.process(detected_face)
					if results.multi_face_landmarks is not None:
						for face_landmarks in results.multi_face_landmarks:
							landmark_list = calc_landmark_list(detected_face, face_landmarks)
							pre_processed_landmark_list = pre_process_landmark(
								landmark_list)
							facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)
							predictions= keypoint_classifier_labels[facial_emotion_id]
							(wt, ht), _ = cv2.getTextSize(predictions, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
							cv2.rectangle(debug_image,(x,y-40),(x+wt,y),(0,0,0),-1)
							cv2.putText(debug_image, predictions, (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
       
							for i in range(0,478):
								pt1=face_landmarks.landmark[i]
								locx=int(pt1.x*image_height)
								locy=int(pt1.y*image_width)
								#cv2.circle(debug_image,(locx,locy),2,(100,100,0),-1)
								#cv2.rectangle(debug_image,(locx,locy),2,(255,255,255),-1)	
								cv2.circle(debug_image,(locx,locy),2,(68,42,32),-1)
								



		return av.VideoFrame.from_ndarray(debug_image, format='bgr24')

webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
				rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
	)