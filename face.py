import os
import secrets
import time
from pathlib import Path

import cv2
from cv2.data import haarcascades

hcc = cv2.CascadeClassifier(haarcascades + 'haarcascade_frontalface_default.xml')


def record(source=0, seconds: float = 5.0, num_pics: int = 25, enforce_num: bool = False) -> Path:
    """
    Captures and detects faces, then saves in a
    :param source: source of video (camera by default)
    :param seconds: capture length
    :param num_pics: minimum number of images
    :param enforce_num: enforce the minimum number of images
    :return: directory hex
    """
    cap = cv2.VideoCapture(source)

    start_time = time.time()
    img_count = 0
    db_dir = secrets.token_hex(16)

    while cap.isOpened() and (time.time() - start_time) < seconds:
        ret, frame = cap.read()
        faces = hcc.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=6)

        if len(faces) > 1:
            print("WARNING: Multiple faces detected!")
            continue

        os.makedirs(f"./db/{db_dir}", exist_ok=True)

        for (x, y, w, h) in faces:
            cv2.imwrite(f"./db/{db_dir}/{img_count}.jpg", frame[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            img_count += 1

        cv2.imshow("Camera", frame)
        if cv2.waitKey(25) == ord('q'):
            break

    if img_count < num_pics:
        print(f"Number of faces captured less than {num_pics=}")

        if enforce_num is True:
            print("Resuming capture...")
            while cap.isOpened() and img_count <= num_pics:
                ret, frame = cap.read()
                faces = hcc.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=6)

                if len(faces) > 1:
                    print("WARNING: Multiple faces detected!")
                    continue

                for (x, y, w, h) in faces:
                    cv2.imwrite(f"./db/{db_dir}/{img_count}.jpg", frame[y:y + h, x:x + w])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    img_count += 1

                cv2.imshow("Camera", frame)
                if cv2.waitKey(25) == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
    return Path(f'./db') / db_dir


if __name__ == '__main__':
    print(record(seconds=10.0, enforce_num=True))
