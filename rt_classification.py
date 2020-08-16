import os
import sys
import cv2
import numpy as np
from data import DataSet
from extractor import Extractor
from keras.models import load_model
from imutils.video import VideoStream

if (len(sys.argv) == 7):
    seq_length = int(sys.argv[1])
    class_limit = int(sys.argv[2])
    height = int(sys.argv[3])
    width = int(sys.argv[4])
    saved_model = sys.argv[5]
    video_file = sys.argv[6]
else:
    seq_length = 50
    class_limit = 2
    height = 120 # 120 # 240 # 480
    width = 160 # 160 # 320 # 640
    saved_model = 'model_021_0.002.hdf5'
    video_file = 'adl-39-cam0.mp4'

capture = cv2.VideoCapture(os.path.join(video_file))
# capture = VideoStream().start()
video_name = video_file.split('/')[-1]
video_name = video_name.split('.')[0]
output_dir = "output_video"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
video_name = "output_video/" + video_name + ".avi"
video_writer = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc('M','J','P','G'), 5, (width,height))

# Get the dataset.
data = DataSet(seq_length=seq_length, class_limit=class_limit, image_shape=(height, width, 3))

# get the model.
extract_model = Extractor(image_shape=(height, width, 3))
saved_LSTM_model = load_model(saved_model)

# frames = []
sequence = []
frame_count = 1
while True:
    ret, frame = capture.read()
    # Bail out when the video file ends
    if not ret:
        break
    # frame = capture.read()

    # Save each frame of the video to a list
    frame = cv2.resize(frame, (width, height))
    # frames.append(frame)

    features = extract_model.extract_image(frame)
    sequence.append(features)

    if frame_count <= seq_length:
        frame_count += 1
        video_writer.write(frame)
        continue
    else:
        sequence.pop(0)

    # Clasify sequence
    prediction = saved_LSTM_model.predict(np.expand_dims(sequence, axis=0))
    print(prediction)
    values = data.print_class_from_prediction(np.squeeze(prediction, axis=0))

    # final_prediction = values[0].split(":")[0]
    # if final_prediction=="Fall":
    #     print("\nFALL\nFALL\nFALL\nFALL\nFALL\n")
    #     break

    # Add prediction to frames and write them to new video
    for i in range(len(values)):
        cv2.putText(frame, values[i], (40, 40 * i + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
    # print("\n\nwriting")
    video_writer.write(frame)
    # cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


video_writer.release()

# python3 rt_classification.py 99 2 180 480 'model_021_0.002.hdf5' 'adl-39-cam0.mp4'
