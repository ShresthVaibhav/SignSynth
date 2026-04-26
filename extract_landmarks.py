import os
import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print('Downloading hand landmarker model...')
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        MODEL_PATH
    )
    print('Model downloaded.')

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

DATASET_PATH = 'dataset/asl_alphabet_train/asl_alphabet_train'
OUTPUT_CSV = 'landmarks.csv'

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)

    header = ['label']
    for i in range(21):
        header += [f'x{i}', f'y{i}', f'z{i}']
    writer.writerow(header)

    labels = os.listdir(DATASET_PATH)

    for label in labels:
        label_path = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(label_path):
            continue

        print(f'Processing {label}...')
        count = 0

        for img_file in os.listdir(label_path):
            img_path = os.path.join(DATASET_PATH, label, img_file)
            mp_image = mp.Image.create_from_file(img_path)
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                row = [label]
                for lm in landmarks:
                    row += [lm.x, lm.y, lm.z]
                writer.writerow(row)
                count += 1

        print(f'{label}: {count} images processed')

print('Done! landmarks.csv created.')