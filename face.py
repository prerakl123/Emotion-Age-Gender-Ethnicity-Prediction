import os
import secrets
import time

import cv2
from cv2.data import haarcascades

hcc = cv2.CascadeClassifier(haarcascades + 'haarcascade_frontalface_default.xml')


def record(seconds: float = 5.0):
    cap = cv2.VideoCapture(0)

    start_time = time.time()
    img_count = 0
    db_dir = secrets.token_hex(16)

    while cap.isOpened() and (time.time() - start_time) < seconds:
        ret, frame = cap.read()
        faces = hcc.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=6)

        os.makedirs(f"./db/{db_dir}", exist_ok=True)

        for (x, y, w, h) in faces:
            cv2.imwrite(f"./db/{db_dir}/{img_count}.jpg", frame[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            img_count += 1

        cv2.imshow("Camera", frame)
        if cv2.waitKey(25) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


record(50.0)
