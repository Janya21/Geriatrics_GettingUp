import os
import sys
import cv2
import numpy as np
from data import DataSet
from extractor import Extractor
from keras.models import load_model
from imutils.video import VideoStream

if (len(sys.argv) == 6):
    seq_length = int(sys.argv[1])
    class_limit = int(sys.argv[2])
    height = int(sys.argv[3])
    width = int(sys.argv[4])
    saved_model = sys.argv[5]
else:
    seq_length = 40
    class_limit = 2
    height = 120 # 120 # 240 # 480
    width = 160 # 160 # 320 # 640
    saved_model = 'model_021_0.002.hdf5'

data_location = "Video/test/Fall"
videos = []
for file in os.listdir(data_location):
    if file.endswith(".mp4"):
        videos.append(file)

videos.sort()

extract_model = Extractor(image_shape=(height, width, 3))
saved_LSTM_model = load_model(saved_model)
data = DataSet(seq_length=seq_length, class_limit=class_limit, image_shape=(height, width, 3))

videos_classified_correctly = []
Correct_Predictions = 0

for video in videos:
    video_file = os.path.join(data_location, video)
    capture = cv2.VideoCapture(os.path.join(video_file))
    sequence = []
    frame_count = 1
    highest_fall = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        features = extract_model.extract_image(frame)
        sequence.append(features)

        if frame_count <= seq_length:
            frame_count += 1
            continue
        else:
            sequence.pop(0)

        # Clasify sequence
        prediction = saved_LSTM_model.predict(np.expand_dims(sequence, axis=0))
        # print(prediction)
        values = data.print_class_from_prediction(np.squeeze(prediction, axis=0))

        final_prediction_state = values[0].split(":")[0]
        final_prediction_val = int (float(values[0].split(":")[1])*100)
        if final_prediction_state=="Fall":
            if final_prediction_val > highest_fall:
              highest_fall = final_prediction_val

    if highest_fall>95:
        Correct_Predictions += 1
        videos_classified_correctly.append(video)
        print(f'{video:13} ==>   Fall             |   Highest Fall Percentage :', highest_fall)
    else:
        print(f'{video:12} ==>   Regular_Activity   |   Highest Fall Percentage :', highest_fall)


print(Correct_Predictions/len(videos))
videos_classified_correctly = set(videos_classified_correctly)
videos = set(videos)
videos_classified_wrongly = videos.difference(videos_classified_correctly)
print("\n\n\n")
print("Number Of Videos Classified Wrongly :", len(videos_classified_wrongly))
print("Total Number Of Videos              :", len(videos))
print("\nVIDEOS CLASSIFIED WRONGLY:\n")
for video in videos_classified_wrongly:
    print(video)
print("\n\n")
# python3 rt_classification.py 99 2 180 480 'model_021_0.002.hdf5' 'adl-39-cam0.mp4'
