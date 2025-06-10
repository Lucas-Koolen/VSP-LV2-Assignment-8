import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# 📂 Laad data
data = np.load("models/training_data.npz")
X, y = data['X'], data['y']

# ⚖️ Normaliseren (voor betere training)
max_x = np.max(X[:, :, 0])
max_y = np.max(X[:, :, 1])
X[:, :, 0] /= max_x
X[:, :, 1] /= max_y
y[:, 0] /= max_x
y[:, 1] /= max_y

# 🔨 Bouw model
model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(X.shape[1], 2)))
model.add(Dense(32, activation='relu'))
model.add(Dense(2))  # x en y

model.compile(optimizer='adam', loss='mse')

# 💾 Opslagpad
os.makedirs("models", exist_ok=True)
checkpoint = ModelCheckpoint("models/lstm_model.keras", save_best_only=True, monitor="loss", mode="min")

# 🚀 Train
model.fit(X, y, epochs=100, batch_size=32, verbose=1, callbacks=[checkpoint])

print("[✅] Model getraind en opgeslagen als: models/lstm_model.keras")
