# 🎯 Real-time Object Tracking & Trajectory Prediction

Dit project trackt een specifiek object op kleur (zoals een paars object) en voorspelt realtime de volgende positie met behulp van een LSTM-neuraal netwerk.

---

## Installatie

1. Zorg dat je Python 3.10 of nieuwer hebt.
2. Installeer de vereisten:

   pip install -r requirements.txt

---

## 📁 Projectstructuur

- `datasets/` – Bevat sessies met trackingvideo’s en CSV-data
  - `session_YYYYMMDD_HHMMSS/`
    - `clip_001.csv`
    - `clip_001.mp4`
- `models/` – Bevat getrainde modellen en gegenereerde trainingsdata
  - `lstm_model.keras`
  - `training_data.npz`
- `.gitignore` – Negeert o.a. datasets/ en models/
- `README.md` – Uitleg en installatie
- `requirements.txt` – Pip dependencies
- `record_tracking_data.py` – Opnemen van trackingclips via webcam
- `prepare_dataset.py` – Zet CSV trackingdata om naar sequenties
- `train_lstm.py` – Traineert of finetunet het LSTM-model
- `predict_live.py` – Live tracking en voorspelling met webcam

---

## Stap 1 – Objecttracking opnemen

Start het script:

    python record_tracking_data.py

- Klik op het object in het camerabeeld (kleurinstelling).
- Druk `1` om opname te starten.
- Druk `2` om opname te stoppen.
- Elke clip wordt automatisch in een nieuwe map geplaatst.

---

## Stap 2 – Dataset voorbereiden

Zet CSV-data om naar trainingsformaat:

    python prepare_dataset.py

Dit genereert `models/training_data.npz`.

---

## Stap 3 – Model trainen

**Nieuw model trainen vanaf nul:**

    python train_lstm.py

**Bestaand model verder trainen:**

Als `lstm_model.keras` al bestaat, wordt deze automatisch geladen.  
Wil je versies bewaren? Gebruik bijvoorbeeld:

- `lstm_model_v1.keras`
- `lstm_model_finetuned.keras`

---

## Stap 4 – Live voorspelling met webcam

Start live tracking + voorspelling:

    python predict_live.py

- Webcambeeld is gespiegeld.
- Rode stip = actuele positie, paarse stip = voorspelling.
- Sluiten met `ESC`.

---

© 2025 – VSP Level 2 Assignment 
