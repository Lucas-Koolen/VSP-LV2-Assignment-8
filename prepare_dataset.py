import os
import csv
import numpy as np

INPUT_DIR = "datasets"
OUTPUT_FILE = "models/training_data.npz"

sequence_length = 5  # aantal frames input → predictie frame

X = []
y = []

def read_csv(filepath):
    coords = []
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # header skippen
        for row in reader:
            x, y = int(row[1]), int(row[2])
            if x == -1 or y == -1:
                continue  # skip frames zonder detectie
            coords.append([x, y])
    return coords

# Doorloop alle sessies en clips
for session in os.listdir(INPUT_DIR):
    session_path = os.path.join(INPUT_DIR, session)
    if not os.path.isdir(session_path):
        continue
    for clip in os.listdir(session_path):
        clip_path = os.path.join(session_path, clip)
        if clip.endswith(".csv"):
            coords = read_csv(clip_path)
        elif os.path.isdir(clip_path):
            for file in os.listdir(clip_path):
                if file.endswith(".csv"):
                    coords = read_csv(os.path.join(clip_path, file))
                else:
                    continue
        else:
            continue

        # Genereer sequenties voor training
        for i in range(len(coords) - sequence_length):
            X.append(coords[i:i+sequence_length])
            y.append(coords[i+sequence_length])

print(f"[INFO] Sequenties gevonden: {len(X)}")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

os.makedirs("models", exist_ok=True)
np.savez_compressed(OUTPUT_FILE, X=X, y=y)
print(f"[✅] Dataset opgeslagen als: {OUTPUT_FILE}")
