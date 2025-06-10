# Real-time Object Tracking & Trajectory Prediction (Purple Cylinder)

Dit project trackt een specifiek object op basis van kleur (zoals een paarse lippenbalsem) en voorspelt de volgende positie realtime met behulp van een LSTM-neuraal netwerk.

--

📦 INSTALLATIE

1. Zorg dat je Python 3.10 of nieuwer hebt geïnstalleerd.
2. Installeer alle afhankelijkheden:

   pip install -r requirements.txt

--

📁 STRUCTUUR VAN DE PROJECTMAP

VSP-LV2-Assignment-8/
├── datasets/                 ← Bevat trackingvideo’s + csv-data per sessie
│   └── session_YYYYMMDD_HHMMSS/
│       └── clip_001/
│           ├── clip_001.csv
│           └── clip_001.mp4
├── models/
│   ├── lstm_model.keras      ← Getraind model (optioneel)
│   └── training_data.npz     ← Sequentiedata voor training
├── record_tracking_data.py   ← Object volgen en trackingdata opnemen
├── prepare_dataset.py        ← Zet .csv trackingdata om naar X/y sequenties
├── train_lstm.py             ← Traineert of finetunet het LSTM-model
├── predict_live.py           ← Live tracking + voorspelling met webcam
├── requirements.txt
└── README.md

--

🔴 STAP 1 – OBJECT TRACKING OPNEMEN

1. Start:  python record_tracking_data.py
2. Klik op het paarse object in het camerabeeld.
3. Druk op:
   - '1' om de opname te starten
   - '2' om de opname te stoppen
4. Elke clip wordt automatisch in een nieuwe map opgeslagen.

Je vindt de .mp4 en bijbehorende .csv per clip onder: datasets/session_xxx/clip_xxx/

--

🟡 STAP 2 – DATA PREPROCESSING

Voer uit:

   python prepare_dataset.py

Dit script scant alle .csv’s en maakt de sequentie-dataset `models/training_data.npz`.

Let op: dit bestand wordt overschreven bij opnieuw uitvoeren.

--

🟢 STAP 3 – LSTM-MODEL TRAINEN OF FINETUNEN

Kies één van de twee opties:

**A: Opnieuw trainen vanaf 0 (scratch):**

1. Verwijder (of hernoem) eventueel het bestaande `models/lstm_model.keras`
2. Voer uit:

   python train_lstm.py

Het script laadt `training_data.npz` en maakt een nieuw model aan.

**B: Verder trainen op bestaand model:**

Het script detecteert automatisch of er al een model bestaat en laadt dat in.

Je kunt dus gewoon `python train_lstm.py` uitvoeren en hij gaat verder met trainen op basis van het bestaande model.

Je kunt eventueel meerdere modellen opslaan met andere namen zoals:

- lstm_model_v1.keras
- lstm_model_finetuned.keras
- lstm_model_sessie2.keras

Pas in `train_lstm.py` de bestandsnaam aan als je een specifiek model wilt herladen of opslaan.

--

🔵 STAP 4 – LIVE VOORSPELLEN MET WEBCAM

Voer uit:

   python predict_live.py

- Het webcambeeld verschijnt, gespiegeld.
- Je ziet:
  - de huidige positie van het object (blauwe stip)
  - de voorspelde volgende positie (paarse stip)
- Sluiten: druk op ESC

Zorg dat het object goed zichtbaar blijft en dat het trackinggebied duidelijk genoeg is voor het model.

--

🛠 TIPS

- Hertrain regelmatig met nieuwe clips als je model niet accuraat genoeg is.
- Zorg voor variatie in je trainingsmateriaal (beweging, afstand, snelheid).
- Voor betere prestaties kun je `train_lstm.py` aanpassen naar meer epochs, andere modelarchitectuur, etc.

--

© 2025 – VSP LVL 2 ASSIGMENT 8
