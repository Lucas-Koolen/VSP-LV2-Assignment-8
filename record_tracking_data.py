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
clip_counter = 1

# 📁 Sessiepad (blijft gelijk zolang programma draait)
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
session_root = os.path.join("datasets", f"session_{now}")
os.makedirs(session_root, exist_ok=True)

# Klik om kleur te selecteren
def select_color(event, x, y, flags, param):
    global target_lab
    if event == cv2.EVENT_LBUTTONDOWN:
        lab_img = cv2.cvtColor(param, cv2.COLOR_BGR2LAB)
        x0, y0 = max(x - 3, 0), max(y - 3, 0)
        x1, y1 = min(x + 3, lab_img.shape[1]), min(y + 3, lab_img.shape[0])
        roi = lab_img[y0:y1, x0:x1]
        avg_lab = np.mean(roi.reshape(-1, 3), axis=0).astype(np.uint8)
        target_lab = avg_lab
        print(f"[INFO] LAB-kleur (gemiddeld): {target_lab}")

# Webcam openen
cap = cv2.VideoCapture(0)
cv2.namedWindow("Tracking View")
print("[INFO] Klik op het paarse object om kleur te kiezen.")
print("[INFO] Druk op 1 om opname te starten, 2 om te stoppen.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    if target_lab is not None:
        diff = lab.astype("float32") - target_lab.astype("float32")
        delta = np.sqrt(np.sum(diff ** 2, axis=2))
        mask = np.uint8((delta < 30).astype(np.uint8)) * 255

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0
        x_out, y_out = -1, -1

        for c in contours:
            area = cv2.contourArea(c)
            if area < 300:
                continue
            rect = cv2.minAreaRect(c)
            width, height = rect[1]
            if width == 0 or height == 0:
                continue
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio < 1.2 or aspect_ratio > 6.0:
                continue
            if area > best_score:
                best = c
                best_score = area

        if best is not None:
            (x, y), radius = cv2.minEnclosingCircle(best)
            if radius > 5:
                x_out, y_out = int(x), int(y)
                cv2.circle(frame, (x_out, y_out), int(radius), (255, 0, 255), 2)
                cv2.putText(frame, f"({x_out}, {y_out})", (x_out+10, y_out),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        if recording:
            out.write(frame)
            csv_writer.writerow([frame_id, x_out, y_out])
            frame_id += 1
    else:
        mask = np.zeros_like(frame[:, :, 0])

    # Display
    cv2.imshow("Tracking View", frame)
    cv2.imshow("Mask", mask)
    cv2.setMouseCallback("Tracking View", select_color, param=frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1') and not recording:
        # Start nieuwe clipmap
        clip_dir = os.path.join(session_root, f"clip_{clip_counter:03d}")
        os.makedirs(clip_dir, exist_ok=True)

        csv_path = os.path.join(clip_dir, f"clip_{clip_counter:03d}.csv")
        video_path = os.path.join(clip_dir, f"clip_{clip_counter:03d}.mp4")

        print(f"[▶️] Start opname: clip_{clip_counter:03d}")
        recording = True
        frame_id = 0

        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'x', 'y'])

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

    elif key == ord('2') and recording:
        print(f"[⏹️] Opname gestopt: clip_{clip_counter:03d}")
        recording = False
        clip_counter += 1
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
