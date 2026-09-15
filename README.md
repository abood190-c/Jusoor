# Jusoor — Sign Language Model Development

A graduation project that develops bidirectional sign language translation models, bridging communication through machine learning. The repository contains two independent model implementations that evolved over the course of the project.

---

## Repository Structure

```
jusoor-model-development/
│
├── FirstModel/       # Static ASL fingerspelling — MLP on hand landmarks
└── SecondModel/      # Dynamic word-level ASL — LSTM on holistic landmark sequences
```

---

## FirstModel — Static ASL Fingerspelling (MLP)

Translates **static hand shapes ↔ letters** in real time. The system is fully bidirectional:

- **Sign → Text**: Camera feed → MediaPipe hand landmarks → MLP classifier → letter/word output
- **Text → Sign**: Input text → canonical pose lookup → animated 3-D hand avatar

### Architecture

| Property | Value |
|---|---|
| Task | Static ASL letter classification |
| Input | 63 normalized hand landmark coordinates (21 landmarks × XYZ) |
| Model | Multi-Layer Perceptron (MLP) |
| Dataset | [ASL Alphabet — Kaggle (grassknoted)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — 87,000 images, 29 classes |
| Test Accuracy | ~99% |
| Landmark detector | MediaPipe Hand Landmarker (`hand_landmarker.task`) |

### Key Scripts

| File | Purpose |
|---|---|
| `step1_extract_landmarks.py` | Extract MediaPipe landmarks from dataset images → CSV |
| `step2_train_mlp.py` | Train MLP classifier on extracted landmarks |
| `step3_live_detector.py` | Real-time sign → text via webcam |
| `step4a_compute_poses.py` | Build canonical pose library for text → sign |
| `step4b_avatar.py` | Drive 3-D hand avatar from pose library |
| `demo.py` | Full bidirectional demo interface |
| `hand_avatar_v13.html` | Browser-based 3-D avatar (Three.js) — text-to-sign output |
| `ws_server.py` | WebSocket server bridging Python backend ↔ avatar frontend |
| `diagnose_skips.py` | Utility: analyse images skipped during landmark extraction |
| `fix_nothing_class.py` | Utility: balance the "nothing" class in the dataset |
| `mirror_augmentation.py` | Utility: augment dataset with mirrored (right-hand) samples |

### Setup

```bash
cd FirstModel
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the ASL Alphabet dataset from Kaggle and place it in a `dataset/` folder:
```
https://www.kaggle.com/datasets/grassknoted/asl-alphabet
```

Download the MediaPipe hand landmark model:
```bash
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Run the pipeline:
```bash
python step1_extract_landmarks.py   # Extract landmarks → landmarks.csv
python step2_train_mlp.py           # Train the MLP → sign_mlp_model.keras
python demo.py                      # Launch the full bidirectional demo
```

---

## SecondModel — Dynamic Word-Level ASL (LSTM)

Moves beyond static letters to recognise **complete ASL words** from video sequences by modelling the full motion over time.

### Architecture

| Property | Value |
|---|---|
| Task | Word-level ASL sign classification from video |
| Input | Sequence of holistic landmark vectors — 333 features/frame × 140 frames |
| Model | Bidirectional LSTM |
| Landmark detector | MediaPipe Holistic (`holistic_landmarker.task`) |

**Feature vector per frame (333 features):**

```
 33 pose landmarks    × 3 (x,y,z) =  99
 21 left-hand         × 3         =  63
 21 right-hand        × 3         =  63
 36 face landmarks    × 3         = 108
─────────────────────────────────────────
                              333 features
```

Each video is resampled to a fixed sequence length of **140 frames**, giving a model input shape of `(140, 333)`.

### Dataset Version History

The model was trained across three dataset iterations:

| Version | Dataset | Classes | Notes |
|---|---|---|---|
| v1 | [WLASL](https://dxli94.github.io/WLASL/) (original) | Subset | Obtained from the original WLASL repository |
| v2 | WLASL — extended splits | WLASL-100/300/1000/2000 | Newer WLASL version with predefined splits |
| v3 | [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) | Expanded | Processed via Kaggle — see below |

#### Dataset v3 — ASL Citizen (Kaggle Preprocessing)

Due to local storage constraints, the **ASL Citizen** landmark extraction was performed on a Kaggle notebook:

> 📓 **[ASL Citizen — MediaPipe Holistic Landmark Extraction](https://www.kaggle.com/code/abdalrhmanashqar/asl-citizen-holistic-landmark-extraction)**

The preprocessing pipeline:

1. Loads the ASL Citizen metadata splits (`train.csv`, `val.csv`, `test.csv`).
2. Runs **MediaPipe Holistic** on each video to extract landmarks frame-by-frame.
3. Concatenates landmarks in `Pose → Left Hand → Right Hand → Face` order (333 features/frame).
4. Resamples each video's landmark sequence to **140 frames** via interpolation.
5. Saves each video as a `.npy` file organised by gloss/class:
   ```
   landmark_dir_v3/
   ├── class_1/
   │   ├── video_1.npy
   │   └── ...
   └── class_2/
       └── ...
   ```
6. Compresses `landmark_dir_v3/` into a ZIP archive for download and use by the training pipeline.

> **Note:** This step is purely a preprocessing pipeline. It does not train the model, classify signs, or modify the original video files.
>
> The `v2 → v3` change is a **dataset change** only — the landmark feature ordering and extraction method remain identical between versions.

### Key Scripts

| File | Purpose |
|---|---|
| `step1_landmark_extraction.py` | Extract MediaPipe Holistic landmarks from video dataset → `.npy` files |
| `step2_training.py` | Train Bi-LSTM classifier on extracted landmark sequences |
| `testing_.py` | Evaluate trained model — produces confusion matrix & accuracy plots |
| `consumers2.py` | Django Channels WebSocket consumer for live inference |
| `frame_counter.py` | Utility: count frames across dataset videos |
| `pipeline/` | Reusable inference pipeline package (see `integration_notes.md`) |

### Label Maps

| File | Description |
|---|---|
| `label_map.json` | v1 class index (20 signs) |
| `label_map_v2.json` | v2 class index |
| `label_map_v3.json` | v3 class index (ASL Citizen classes) |

### Backend Integration (Django)

The model is served via a Django Channels WebSocket. See [`integration_notes.md`](SecondModel/integration_notes.md) for the full setup guide including:

- Required `INSTALLED_APPS`, ASGI config, and channel layer settings.
- WebSocket protocol contract (binary JPEG frames in, JSON predictions out).
- Logging configuration and critical deployment notes (Daphne required, Python 3.11).

### Setup

```bash
cd SecondModel
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the MediaPipe Holistic model:
```bash
wget https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
```

Run the pipeline:
```bash
python step1_landmark_extraction.py   # Extract landmarks from dataset videos
python step2_training.py              # Train the Bi-LSTM model
python testing_.py                    # Evaluate and generate confusion matrix
```

---

## What Is and Isn't Tracked by Git

Large binary files are excluded from version control. The table below summarises the policy:

| File type | Tracked? | Reason |
|---|---|---|
| Python source (`.py`) | ✅ Yes | Core deliverable |
| HTML avatar UI (`.html`) | ✅ Yes | Text-to-sign output interface |
| `hand.glb` | ✅ Yes | Required 3-D asset (~1 MB) |
| Label maps (`*.json`) | ✅ Yes | Needed for inference without retraining |
| Training charts (`*.png`) | ✅ Yes | Results documentation |
| `requirements.txt` | ✅ Yes | Reproducibility |
| Dataset folders (`Dataset*/`) | ❌ No | Too large — use Kaggle/WLASL links |
| Landmark arrays (`*.npy`, `*.csv`) | ❌ No | Regenerated by extraction scripts |
| Model weights (`*.keras`, `*.pkl`) | ❌ No | Regenerated by training |
| MediaPipe models (`*.task`) | ❌ No | Downloaded via `wget` — see setup above |
| Virtual environments | ❌ No | Recreated via `pip install -r requirements.txt` |

---

## Dataset Credits

- [ASL Alphabet Dataset — grassknoted on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [WLASL — Word-Level ASL Dataset](https://dxli94.github.io/WLASL/)
- [ASL Citizen — Microsoft Research](https://www.microsoft.com/en-us/research/project/asl-citizen/)
