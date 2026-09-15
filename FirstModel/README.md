# Jusoor — Bidirectional ASL Fingerspelling System

A real-time bidirectional American Sign Language (ASL) fingerspelling translation system built as a graduation project. The system translates live hand gestures into text, and text back into animated sign language — designed with extensibility in mind for future support of other sign languages and dialects.

---

## How It Works

The pipeline has two directions:

**Sign → Text**
1. Camera captures live hand video
2. MediaPipe detects 21 hand landmarks per frame
3. Landmarks are normalized (position & scale invariant)
4. An MLP classifier predicts the ASL letter
5. A smoothing buffer stabilizes predictions across frames
6. Letters accumulate into words

**Text → Sign**
1. User types a word or sentence
2. Each letter is looked up in a canonical pose library
3. An avatar animates through the corresponding hand positions

---

## Project Structure

```
ProjectFolder/
│
├── extract_landmarks.py      # Extract MediaPipe landmarks from dataset images → CSV
├── train_mlp.py              # Train MLP classifier on extracted landmarks
├── live_detection.py         # Real-time sign → text translation via webcam
├── text_to_sign.py           # Text → sign avatar animation
├── demo.py                   # Full bidirectional demo interface
│
├── diagnose_skips.py         # Utility: analyze skipped images during extraction
├── fix_nothing_class.py      # Utility: fix/balance the "nothing" class in dataset
├── mirror_augmentation.py    # Utility: augment dataset with mirrored (right-hand) samples
│
└── requirements.txt          # Python dependencies
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| Dataset | ASL Alphabet (Kaggle) — 87,000 images, 29 classes |
| Test Accuracy | 99.09% |
| Architecture | MLP (Multi-Layer Perceptron) |
| Input Features | 63 normalized hand landmark coordinates |

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/abood190-c/Model.git
cd Model
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the ASL Alphabet dataset from Kaggle:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Place it in a folder named `dataset/` in the project root.

### 4. Download the MediaPipe hand landmark model
```bash
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 5. Run the pipeline

**Step 1 — Extract landmarks from dataset:**
```bash
python extract_landmarks.py
```

**Step 2 — Train the model:**
```bash
python train_mlp.py
```

**Step 3 — Run the live demo:**
```bash
python demo.py
```

---

## Architecture Design

The system is designed to be extensible:
- The landmark normalization and MLP classifier are language-agnostic
- Adding support for a new sign language requires a new dataset and retraining
- Future work includes LSTM-based dynamic sign recognition (WLASL dataset) for full word-level translation

---

## Dependencies

See `requirements.txt`. Main libraries:
- `mediapipe` — hand landmark detection
- `opencv-python` — camera feed and image processing
- `tensorflow` / `keras` — MLP model
- `scikit-learn` — preprocessing and evaluation
- `numpy`, `pandas` — data handling

---

## Dataset Credit

[ASL Alphabet Dataset by grassknoted on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
