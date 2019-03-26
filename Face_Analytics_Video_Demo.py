#!/usr/bin/env python
# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import cv2
from statistics import mode
from keras.models import load_model
import numpy as np
import time
# USER DEFINED METHODS
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from utils.wide_resnet import WideResNet

# parameters for loading data and images
print("[INFO] Setting Required Parameters for the demo...")

detection_model_path = 'models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = 'models/gender_models/simple_CNN.81-0.96.hdf5'
age_model_path = 'models/age_models/weights.28-3.73.hdf5'

emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
gender_labels = {0: 'Female', 1: 'Male'}
age_labels = ['Children', 'Adolescents', 'Adults', 'Middle Age', 'Old', 'Very Old']
font = cv2.FONT_HERSHEY_SIMPLEX

# load the known faces and embeddings
data = pickle.loads(open("Models/face_encodings", "rb").read())

# hyper-parameters for bounding boxes shape
frame_window = 10

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# load model and weights
img_size = 64
model = WideResNet(img_size, depth=16, k=8)()
model.load_weights(age_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# starting lists for calculating modes
gender_window = []
emotion_window = []

# starting video streaming
cv2.namedWindow('Face Analysis')
video_capture = VideoStream().start()
writer = None
time.sleep(2.0)

while True:
    bgr_image = video_capture.read()
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    # detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb_image, model='hog')
    encodings = face_recognition.face_encodings(rgb_image, boxes)
    names = []
    
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "UNK"
        unknown_counter = 1
        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a dictionary to count the total number of 
            # times each face was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for each recognized face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)
        else:
            name = "UNK" + "-" + str(unknown_counter)
            unknown_counter = unknown_counter + 1
                
        # update the list of names
        names.append(name)
         
    for ((top, right, bottom, left), name, face_coordinates) in zip(boxes, names, faces):
        x1 = left - 20
        x2 = right + 20
        y1 = left - 40
        y2 = bottom + 40
        rgb_face = rgb_image[y1:y2, x1:x2]
        gray_face = gray_image[y1:y2, x1:x2]

        try:
            age_rgb_face = cv2.resize(rgb_face, (emotion_target_size))
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, False)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        rgb_face = np.expand_dims(rgb_face, 0)
        rgb_face = preprocess_input(rgb_face, False)
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]
        gender_window.append(gender_text)

        age_rgb_face = np.expand_dims(age_rgb_face, 0)
        age_prediction = model.predict(age_rgb_face)
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = age_prediction[1].dot(ages).flatten()
        if predicted_ages <= 9.5:
            age_group = age_labels[0]
        elif predicted_ages <= 19.5:
            age_group = age_labels[1]
        elif predicted_ages <= 45.5:
            age_group = age_labels[2]
        elif predicted_ages <= 60:
            age_group = age_labels[3]
        elif predicted_ages <= 80:
            age_group = age_labels[4]
        else:
            age_group = age_labels[5]

        if len(gender_window) > frame_window:
            emotion_window.pop(0)
            gender_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
            gender_mode = mode(gender_window)
        except:
            continue
        
        if gender_text == gender_labels[0]:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        draw_bounding_box(np.array([left, top, right - left + 10, bottom - top + 10]), rgb_image, color)

        draw_text(np.array([left, top, right, bottom]),
                  rgb_image, gender_mode + " | " + emotion_mode + " | " + age_group, (0, 255, 0), 0, -5, 0.5, 1)

        draw_text(np.array([left, top, right, bottom], dtype="int32"),
                  rgb_image, name, (0, 255, 0), 0, -20, 0.75, 2)
        
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    if 1 > 0:
        cv2.imshow("Face Analysis", bgr_image)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
cv2.destroyAllWindows()
video_capture.stop()

# check to see if the video writer point needs toq be released
if writer is not None:
    writer.release()



