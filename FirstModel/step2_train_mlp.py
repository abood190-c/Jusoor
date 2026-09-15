"""
STEP 3 & 4: MLP TRAINING
=========================
What this script does:
    Loads the landmarks CSV produced by step1, builds an MLP classifier,
    trains it with early stopping and learning rate scheduling, evaluates
    it on a held-out test set, and saves the model + label encoder.

Why MLP over other options:
    - Input is 63 structured numbers, not an image — no spatial patterns
      to convolve over, so CNN is the wrong tool here.
    - Compared to SVM/Random Forest: natural upgrade path to LSTM for
      dynamic signs later (just prepend sequence input to the dense layers).
    - Fast to train, easy to deploy, good accuracy on landmark features.

Usage:
    python step2_train_mlp.py
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import LabelEncoder
from sklearn.metrics            import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CSV_PATH       = "landmarks.csv"
MODEL_PATH     = "sign_mlp_model.keras"
ENCODER_PATH   = "label_encoder.pkl"  # we must save this alongside the model

# Training hyperparameters
BATCH_SIZE     = 64
MAX_EPOCHS     = 200    # early stopping will likely kick in before this
LEARNING_RATE  = 1e-3
VAL_SPLIT      = 0.15   # 15% of training data for validation
TEST_SPLIT     = 0.15   # 15% for final held-out test evaluation


# ─────────────────────────────────────────────
# STEP A: LOAD & PREPARE DATA
# ─────────────────────────────────────────────

print("Loading landmarks CSV...")
df = pd.read_csv(CSV_PATH)

print(f"  Total samples: {len(df)}")
print(f"  Classes: {sorted(df['label'].unique())}")
print(f"  Class distribution:\n{df['label'].value_counts().sort_index()}\n")

# Separate features and labels
# For X the 63 landmarks were converted to float32 because that's what PyTorch expects
X = df.drop(columns=["label"]).values.astype(np.float32)
y_raw = df["label"].values

# Label encoding
# Why: Neural networks need integer or one-hot targets not strings.
# The encoder is saved so the detector can map predicted integers back to letters.
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
num_classes = len(encoder.classes_)
print(f"Encoded {num_classes} classes: {list(encoder.classes_)}")

# Save encoder — the detector MUST use the same one
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(encoder, f)
print(f"✓ Label encoder saved to {ENCODER_PATH}")

# Train / Validation / Test split
# Why a separate test set:
#   Validation loss is used by early stopping and therefore influences
#   which model gets saved — it's not a clean measure of generalization.
#   The test set is touched ONLY at the very end to get an honest number.
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size   = TEST_SPLIT,
    stratify    = y,        # ensure each class is proportionally represented
    random_state= 42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size   = VAL_SPLIT / (1 - TEST_SPLIT),
    stratify    = y_trainval,
    random_state= 42
)

print(f"\nSplit sizes:")
print(f"  Train:      {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Test:       {len(X_test)} samples")


# ─────────────────────────────────────────────
# STEP B: BUILD THE MLP
# ─────────────────────────────────────────────
# Architecture reasoning:
#
#   Input(63)
#     → BatchNorm       normalize input distribution, speeds convergence
#     → Dense(256, ReLU)  first feature extraction layer
#     → Dropout(0.3)    randomly zero 30% of neurons during training
#     → Dense(256, ReLU)  second feature extraction layer
#     → Dropout(0.3)
#     → Dense(128, ReLU)  bottleneck — forces compact representation
#     → Dense(num_classes, Softmax)
#
#   Why two 256 layers:
#       Sign language fingerspelling has subtle differences between some
#       letters (e.g. M/N/T). Two layers give the model enough depth to
#       learn these distinctions without being excessively large.
#
#   Why Dropout(0.3):
#       With only 63 input features the model can memorize training data
#       quickly. Dropout forces redundant representations, improving
#       generalization to real webcam input.
#
#   Why BatchNormalization on input:
#       Even after our landmark normalization, different axes (x, y, z)
#       have different variance. BatchNorm standardizes this automatically.

def build_mlp(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.BatchNormalization(),

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(128, activation="relu"),

        layers.Dense(num_classes, activation="softmax"),
    ], name="sign_language_mlp")
    return model


model = build_mlp(X_train.shape[1], num_classes)
model.summary()

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss      = "sparse_categorical_crossentropy",
    metrics   = ["accuracy"]
)


# ─────────────────────────────────────────────
# STEP C: TRAINING CALLBACKS
# ─────────────────────────────────────────────

# Early Stopping
# Why: Stops training when val_loss stops improving, and restores the
# best weights seen during training. Prevents wasting time and overfitting.
early_stop = callbacks.EarlyStopping(
    monitor              = "val_loss",
    patience             = 15,       # wait 15 epochs before giving up
    restore_best_weights = True,     # roll back to the best checkpoint
    verbose              = 1
)

# ReduceLROnPlateau
# Why: When the model plateaus, the current learning rate may be too large
# to make finer adjustments. This halves the LR when val_loss stalls,
# allowing the optimizer to explore more carefully near a minimum.
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor  = "val_loss",
    factor   = 0.5,          # multiply LR by 0.5
    patience = 7,
    min_lr   = 1e-6,
    verbose  = 1
)

# Model Checkpoint
# Why: Saves the best model to disk during training — insurance against
# crashes, and ensures we always have the best version regardless of
# what happens at the end of training.
checkpoint = callbacks.ModelCheckpoint(
    filepath         = MODEL_PATH,
    monitor          = "val_accuracy",
    save_best_only   = True,
    verbose          = 1
)


# ─────────────────────────────────────────────
# STEP D: TRAIN
# ─────────────────────────────────────────────

print("\nStarting training...")

history = model.fit(
    X_train, y_train,
    epochs          = MAX_EPOCHS,
    batch_size      = BATCH_SIZE,
    validation_data = (X_val, y_val),
    callbacks       = [early_stop, reduce_lr, checkpoint],
    verbose         = 1
)

print(f"\n✓ Training complete. Best model saved to {MODEL_PATH}")


# ─────────────────────────────────────────────
# STEP E: EVALUATION ON HELD-OUT TEST SET
# ─────────────────────────────────────────────
# This is the honest accuracy — the model has never seen this data.

print("\n─── Test Set Evaluation ───")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Accuracy: {test_acc * 100:.2f}%")
print(f"  Test Loss:     {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=encoder.classes_
))


# ─────────────────────────────────────────────
# STEP F: PLOT TRAINING CURVES + CONFUSION MATRIX
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(20, 16))

# Accuracy curve
axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

# Loss curve
axes[1].plot(history.history["loss"],     label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

# Confusion matrix
# Why: Overall accuracy can hide per-class weaknesses. The confusion matrix
# shows you exactly which signs get confused with each other (e.g. M/N/T
# are notoriously similar in ASL).
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot      = True,
    fmt        = "d",
    xticklabels= encoder.classes_,
    yticklabels= encoder.classes_,
    ax         = axes[2],
    cmap       = "Blues"
)
axes[2].set_title("Confusion Matrix (Test Set)")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
plt.show()
print("✓ Training plots saved to training_results.png")
