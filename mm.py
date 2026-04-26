import os
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

DATASET_PATH = 'dataset/isl_kaggle'
OUTPUT_CSV = 'isl_landmarks.csv'

LANDMARK_SIZE = 63  # 21 * 3

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)

    header = ['label']
    for hand in ['h1', 'h2']:
        for i in range(21):
            header += [f'{hand}_x{i}', f'{hand}_y{i}', f'{hand}_z{i}']
    writer.writerow(header)

    labels = sorted(os.listdir(DATASET_PATH))

    for label in labels:
        label_path = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(label_path):
            continue

        print(f'Processing {label}...')
        count = 0

        for img_file in os.listdir(label_path):
            img_path = os.path.join(DATASET_PATH, label, img_file)
            try:
                mp_image = mp.Image.create_from_file(img_path)
                result = detector.detect(mp_image)

                row = [label]

                if len(result.hand_landmarks) >= 1:
                    for lm in result.hand_landmarks[0]:
                        row += [lm.x, lm.y, lm.z]
                else:
                    row += [0.0] * LANDMARK_SIZE

                if len(result.hand_landmarks) >= 2:
                    for lm in result.hand_landmarks[1]:
                        row += [lm.x, lm.y, lm.z]
                else:
                    row += [0.0] * LANDMARK_SIZE

                writer.writerow(row)
                count += 1

            except Exception:
                continue

        print(f'{label}: {count} images processed')

print('Done! isl_landmarks.csv created.')