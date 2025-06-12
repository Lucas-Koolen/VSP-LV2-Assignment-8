import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Verberg TensorFlow-logging

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Laad het model
model = load_model('models/lstm_model.keras')

# Parameters
SEQUENCE_LENGTH = 30
HSV_LOWER = np.array([22, 120, 180])
HSV_UPPER = np.array([35, 255, 255])


cap = cv2.VideoCapture(0)
sequence = deque(maxlen=SEQUENCE_LENGTH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    output = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 150:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                sequence.append(center)
                cv2.circle(output, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(output, center, 5, (0, 0, 255), -1)

    if len(sequence) == SEQUENCE_LENGTH:
        input_seq = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 2).astype(np.float32)
        input_seq[:, :, 0] /= 640
        input_seq[:, :, 1] /= 480

        prediction = model.predict(input_seq, verbose=0)[0]
        predicted_x = int(prediction[0] * 640)
        predicted_y = int(prediction[1] * 480)

        cv2.circle(output, (predicted_x, predicted_y), 10, (255, 0, 255), 3)
        cv2.putText(output, 'Voorspelling', (predicted_x + 10, predicted_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    cv2.imshow("Live Tracking & Prediction (Mirrored)", output)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
