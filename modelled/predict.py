import cv2
from cv2 import data
import numpy as np
import tensorflow as tf
from keras.models import load_model

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.utils.disable_interactive_logging()

emotion_model = load_model('./weights/emotion_21-ep_350-lr_1e-03-16042024_215317.h5', compile=True)
age_model = load_model('./weights/age_28-ep_200-lr_0_001-24042024_040756.h5', compile=True)
ethnicity_model = load_model('./weights/ethnicity_28-ep_200-lr_0_001-24042024_041325.h5', compile=True)
gender_model = load_model('./weights/gender_15-ep_200-lr_0_001-24042024_040855.h5', compile=True)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_map = {
    0: 'Angry',
    1: 'Digust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprised',
    6: 'Neutral'
}

age_map = {
    0: "0 - 5",
    1: "5 - 18",
    2: "18 - 30",
    3: "30 - 45",
    4: "45 - 60",
    5: "60 - 80",
    6: "80 - 116"
}

gender_map = {
    1: "Female",
    0: "Male"
}

ethnicity_map = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Hispanic"
}

model_map = {
    0: [[emotion_model], [emotion_map]],
    1: [[age_model], [age_map]],
    2: [[ethnicity_model], [ethnicity_map]],
    3: [[gender_model], [gender_map]],
    -1: [
        [emotion_model, age_model, ethnicity_model, gender_model],
        [emotion_map, age_map, ethnicity_map, gender_map]
    ]
}


display_class = -1


def predict_class(c, face):
    models, maps = model_map[c]
    predictions = []
    for _model, _map in zip(models, maps):
        am = np.argmax(_model.predict(face), axis=-1)
        # print(am)
        try:
            predictions.append(
                _map[am[0]]
            )
        except KeyError:
            predictions.append(
                str(am[0]) + "Unknown"
            )
    # print(predictions)
    return predictions


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi / 255.0
        face_roi = np.reshape(face_roi, (1, 48, 48, 1))

        preds = predict_class(display_class, face_roi)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )
        for i, pred in enumerate(preds):
            cv2.putText(
                frame, pred,
                (x - 75, y + 20 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    cv2.imshow('Emotion Detection', frame)

    key = cv2.waitKey(1)
    if key == 27:  # Escape key
        break

    elif key == ord('m'):  # Display emotion
        display_class = 0

    elif key == ord('a'):  # Display age
        display_class = 1

    elif key == ord('e'):  # Display ethnicity
        display_class = 2

    elif key == ord('g'):  # Display gender
        display_class = 3

    elif key == 32:  # Display all
        display_class = -1

# Release the capture
cap.release()
cv2.destroyAllWindows()
