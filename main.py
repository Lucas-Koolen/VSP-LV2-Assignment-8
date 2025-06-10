# Color Tracking and Trajectory Prediction using OpenCV, Kalman Filter, and Tkinter Sliders with Color Preview and HSV Debug

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

# Define Kalman Filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# Initial HSV ranges
hue = [0, 20]
sat = [100, 255]
val = [100, 255]

# Create Tkinter window with sliders and color preview
def create_slider_window():
    window = tk.Tk()
    window.title("HSV Color Tuner")
    window.geometry("300x400")

    def update_vals():
        hue[0] = int(hue_low.get())
        hue[1] = int(hue_high.get())
        sat[0] = int(sat_low.get())
        sat[1] = int(sat_high.get())
        val[0] = int(val_low.get())
        val[1] = int(val_high.get())

        # Preview the middle HSV value as BGR
        mid_h = int((hue[0] + hue[1]) / 2)
        mid_s = int((sat[0] + sat[1]) / 2)
        mid_v = int((val[0] + val[1]) / 2)

        hsv_color = np.uint8([[[mid_h, mid_s, mid_v]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        hex_color = '#%02x%02x%02x' % (bgr_color[2], bgr_color[1], bgr_color[0])  # RGB for Tkinter

        color_preview.config(bg=hex_color)
        window.after(100, update_vals)

    ttk.Label(window, text="Hue Low").pack()
    hue_low = ttk.Scale(window, from_=0, to=179, orient='horizontal')
    hue_low.set(hue[0])
    hue_low.pack()

    ttk.Label(window, text="Hue High").pack()
    hue_high = ttk.Scale(window, from_=0, to=179, orient='horizontal')
    hue_high.set(hue[1])
    hue_high.pack()

    ttk.Label(window, text="Sat Low").pack()
    sat_low = ttk.Scale(window, from_=0, to=255, orient='horizontal')
    sat_low.set(sat[0])
    sat_low.pack()

    ttk.Label(window, text="Sat High").pack()
    sat_high = ttk.Scale(window, from_=0, to=255, orient='horizontal')
    sat_high.set(sat[1])
    sat_high.pack()

    ttk.Label(window, text="Val Low").pack()
    val_low = ttk.Scale(window, from_=0, to=255, orient='horizontal')
    val_low.set(val[0])
    val_low.pack()

    ttk.Label(window, text="Val High").pack()
    val_high = ttk.Scale(window, from_=0, to=255, orient='horizontal')
    val_high.set(val[1])
    val_high.pack()

    ttk.Label(window, text="Color Preview").pack(pady=5)
    global color_preview
    color_preview = tk.Label(window, text="", width=20, height=2)
    color_preview.pack()

    update_vals()
    return window

slider_window = create_slider_window()

# Start video capture
cap = cv2.VideoCapture(0)
predicted_coords = np.zeros((2, 1), np.float32)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if y < hsv.shape[0] and x < hsv.shape[1]:
            print(f"HSV at ({x}, {y}): {hsv[y, x]}")

cv2.namedWindow("Tracking and Prediction")
cv2.setMouseCallback("Tracking and Prediction", mouse_callback)

while True:
    slider_window.update()

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([hue[0], sat[0], val[0]])
    upper_hsv = np.array([hue[1], sat[1], val[1]])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 5:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            center = np.array([[np.float32(x)], [np.float32(y)]])
            kalman.correct(center)
            predicted = kalman.predict()
            predicted_coords = (int(predicted[0]), int(predicted[1]))
            cv2.circle(frame, predicted_coords, 5, (0, 0, 255), -1)

    cv2.imshow("Tracking and Prediction", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
slider_window.destroy()