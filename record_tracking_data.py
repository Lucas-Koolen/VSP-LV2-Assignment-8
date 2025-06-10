import cv2
import numpy as np
import os
import csv
import datetime

target_lab = None
recording = False
csv_writer = None
csv_file = None
out = None
frame_id = 0

# Start sessie
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
session_name = f"session_{now}"
output_dir = os.path.join("datasets", session_name)
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, f"{session_name}.csv")
video_path = os.path.join(output_dir, f"{session_name}.mp4")

# Klik op het balletje om kleur te kiezen
def select_color(event, x, y, flags, param):
    global target_lab
    if event == cv2.EVENT_LBUTTONDOWN:
        lab_img = cv2.cvtColor(param, cv2.COLOR_BGR2LAB)
        target_lab = lab_img[y, x]
        print(f"[INFO] LAB-kleur ingesteld op: {target_lab}")

# Camera openen
cap = cv2.VideoCapture(0)
cv2.namedWindow("Tracking View")
print("[INFO] Klik op het balletje om de kleur te selecteren.")
print("[INFO] Druk op 1 om de opname te starten, 2 om te stoppen.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    # Alleen masker genereren als we een doelkleur hebben
    if target_lab is not None:
        diff = lab.astype("float32") - target_lab.astype("float32")
        delta = np.linalg.norm(diff, axis=2)
        mask = np.uint8((delta < 45).astype(np.uint8)) * 255

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, None, iterations=2)

        # Contourdetectie
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 200:
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if circularity > 0.6:
                if area > best_score:
                    best = c
                    best_score = area

        x_out, y_out = -1, -1
        if best is not None:
            (x, y), radius = cv2.minEnclosingCircle(best)
            if radius > 5:
                x_out, y_out = int(x), int(y)
                cv2.circle(frame, (x_out, y_out), int(radius), (0, 255, 0), 2)
                cv2.putText(frame, f"({x_out}, {y_out})", (x_out+10, y_out),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Schrijf frame en coördinaten als opname actief is
        if recording:
            out.write(frame)
            csv_writer.writerow([frame_id, x_out, y_out])
            frame_id += 1
    else:
        mask = np.zeros_like(frame[:, :, 0])

    # Toon scherm
    cv2.imshow("Tracking View", frame)
    cv2.imshow("Mask", mask)
    cv2.setMouseCallback("Tracking View", select_color, param=frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1') and not recording:
        # Start opname
        print("[▶️] Opname gestart.")
        recording = True
        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'x', 'y'])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
        frame_id = 0

    elif key == ord('2') and recording:
        # Stop opname
        print("[⏹️] Opname gestopt.")
        recording = False
        if csv_file:
            csv_file.close()
        if out:
            out.release()

    elif key == ord('q') or key == 27:
        print("[🛑] Programma gestopt.")
        break

cap.release()
if out and out.isOpened():
    out.release()
if csv_file and not csv_file.closed:
    csv_file.close()
cv2.destroyAllWindows()
