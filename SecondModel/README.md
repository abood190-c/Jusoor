# SecondModel — Dynamic Word-Level ASL Recognition (Bi-LSTM)

This module extends the static fingerspelling work of [FirstModel](../FirstModel) to recognise **complete ASL words** from video sequences by capturing the full temporal motion of hands, pose, and face.

---

## How It Works

1. A video of a signer is loaded frame-by-frame.
2. **MediaPipe Holistic** extracts pose, hand, and face landmarks from each frame.
3. The landmark sequence is resampled to a fixed length of **140 frames**.
4. A **Bidirectional LSTM** model classifies the sequence into a word/gloss.
5. In the live demo, a Django Channels WebSocket streams frames from the client, runs inference, and returns predictions as JSON.

---

## Feature Vector

333 features are extracted per frame by concatenating landmarks in this fixed order:

```
Pose  (33 × 3) =  99
Left Hand  (21 × 3) =  63
Right Hand (21 × 3) =  63
Face  (36 × 3) = 108
──────────────────────
         333 features/frame
```

Each video is normalised to **(140, 333)** — 140 timesteps × 333 features.

---

## Project Structure

```
SecondModel/
│
├── pipeline/                       # Reusable inference pipeline package
│   ├── __init__.py
│   ├── frame_capturing.py          # Frame capture & buffering
│   ├── frame_to_sequence.py        # Landmark extraction & sequence normalisation
│   ├── model_inference.py          # Model prediction & confidence scoring
│   ├── model_loader.py             # Singleton model loader (thread-safe)
│   └── utils.py
│
├── step1_landmark_extraction.py    # Extract landmarks from dataset videos → .npy files
├── step2_training.py               # Train Bi-LSTM on extracted landmark sequences
├── step3.py                        # (Inference / evaluation script)
├── testing_.py                     # Evaluate model — confusion matrix & accuracy plots
├── consumers2.py                   # Django Channels WebSocket consumer
├── frame_counter.py                # Utility: count frames across dataset videos
├── integration_notes.md            # Full Django backend integration guide
│
├── label_map.json                  # v1 class index (20 signs)
├── label_map_v2.json               # v2 class index
├── label_map_v3.json               # v3 class index (ASL Citizen)
├── cm.json                         # Confusion matrix data (JSON)
│
├── sign_lstm_model.keras           # Trained model — v1  [git-ignored]
├── sign_lstm_model_v2.keras        # Trained model — v2  [git-ignored]
├── sign_lstm_model_v3.keras        # Trained model — v3  [git-ignored]
├── holistic_landmarker.task        # MediaPipe Holistic model [git-ignored]
│
├── Dataset/                        # WLASL v1 videos      [git-ignored]
├── Dataset_v2/                     # WLASL v2 videos      [git-ignored]
├── Dataset_v3/                     # ASL Citizen videos   [git-ignored]
├── landmark_dir/                   # Extracted .npy — v1  [git-ignored]
├── landmark_dir_v2/                # Extracted .npy — v2  [git-ignored]
├── landmark_dir_v3/                # Extracted .npy — v3  [git-ignored]
│
├── accuracy_loss.png               # Training accuracy/loss chart
├── confusion_matrix.png            # Confusion matrix visualisation
├── training_results.png            # Combined training results chart
│
└── requirements.txt                # Python dependencies
```

---

## Dataset Version History

| Version | Dataset | Notes |
|---|---|---|
| **v1** | [WLASL](https://dxli94.github.io/WLASL/) | Original WLASL source |
| **v2** | WLASL — extended | Includes WLASL-100/300/1000/2000 splits |
| **v3** | [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) | Processed via Kaggle due to storage constraints |

### Dataset v3 — ASL Citizen Landmark Extraction

Because the full ASL Citizen dataset exceeds local storage capacity, the landmark extraction for v3 was performed inside a **Kaggle notebook**:

> 📓 **[ASL Citizen — MediaPipe Holistic Landmark Extraction](https://www.kaggle.com/code/abdalrhmanashqar/asl-citizen-holistic-landmark-extraction)**

The pipeline:

1. Loads the ASL Citizen metadata splits (`train.csv`, `val.csv`, `test.csv`).
2. Runs **MediaPipe Holistic** on each video, extracting landmarks frame-by-frame.
3. Concatenates landmarks in `Pose → Left Hand → Right Hand → Face` order (333 features/frame).
4. Resamples each video's landmark sequence to **140 frames** via interpolation.
5. Saves each video as a `.npy` file, organised by gloss/class under `landmark_dir_v3/`.
6. Compresses `landmark_dir_v3/` into a ZIP archive for download and use by the training script.

> **Note:** The `v2 → v3` transition is a **dataset change only**. The landmark feature ordering, extraction method, and input shape (`140 × 333`) are identical between versions.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the MediaPipe Holistic model:
```bash
wget https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
```

---

## Running the Pipeline

```bash
# Step 1 — Extract landmarks from dataset videos
python step1_landmark_extraction.py

# Step 2 — Train the Bi-LSTM model
python step2_training.py

# Step 3 — Evaluate and generate charts
python testing_.py
```

---

## Model Performance

Training result charts are tracked in the repository:

- [`accuracy_loss.png`](accuracy_loss.png) — Training vs. validation accuracy and loss
- [`confusion_matrix.png`](confusion_matrix.png) — Per-class confusion matrix
- [`training_results.png`](training_results.png) — Combined summary

---

## Backend Integration

This model is served over a WebSocket using **Django Channels + Daphne**. The `pipeline/` package handles frame buffering, landmark extraction, and model inference as a thread-safe singleton.

**WebSocket protocol:**
- Client sends: binary JPEG frames
- Server responds: JSON
  ```json
  {"type": "gesture_result", "prediction": "drink", "confidence": 0.94}
  ```

For the full Django settings, routing, and deployment notes, see [`integration_notes.md`](integration_notes.md).

> ⚠️ **Python 3.11 required** — do not use 3.12+. Run with **Daphne**, not `manage.py runserver`.

---

## Dependencies

From `requirements.txt`:

| Library | Version | Purpose |
|---|---|---|
| `mediapipe` | 0.10.9 | Holistic landmark extraction |
| `opencv-contrib-python` | 4.13.0.92 | Video frame processing |
| `numpy` | 2.4.6 | Array operations |
| `scipy` | 1.17.1 | Sequence resampling/interpolation |
| `tensorflow` | 2.21.0 | Bi-LSTM model training & inference |

