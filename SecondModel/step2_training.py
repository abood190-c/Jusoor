
import cv2
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

NUM_CLASSES = None  # Will be set after loading label_map.json
BATCH_SIZE = 32
MAX_FRAMES = 140
LEARNING_RATE = 0.001
MAX_EPOCHS = 60  
MODEL_PATH = "sign_lstm_model_v3.keras"
LANDMARK_DIR = 'landmark_dir_v3'
FILTERED_SPLITS_DIR = "Dataset_v3/filtered_splits"

label_map = json.load(open('label_map_v3.json'))



NUM_CLASSES = len(label_map)
print(f"Loaded {NUM_CLASSES} classes from label_map.json")

# ========================= DATA LOADING =========================
def build_npy_path(row):
    gloss = str(row['Gloss']).strip()
    video_filename = os.path.basename(row['Video file'])
    npy_filename = video_filename.replace('.mp4', '.npy')
    return os.path.join(LANDMARK_DIR, gloss, npy_filename)

train_df = pd.read_csv(os.path.join(FILTERED_SPLITS_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(FILTERED_SPLITS_DIR, "val.csv"))
test_df  = pd.read_csv(os.path.join(FILTERED_SPLITS_DIR, "test.csv"))

for df in [train_df, val_df, test_df]:
    df['npy_path'] = df.apply(build_npy_path, axis=1)
    df['label'] = df['Gloss'].map(label_map)

# Proper filtering
train_df = train_df[train_df['npy_path'].apply(os.path.exists)].reset_index(drop=True)
val_df   = val_df[val_df['npy_path'].apply(os.path.exists)].reset_index(drop=True)
test_df  = test_df[test_df['npy_path'].apply(os.path.exists)].reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} samples")

# ========================= AUGMENTATIONS =========================
def mirror_augmentation(landmarks):
    mirrored = landmarks.copy()
    mirrored[:, 0::3] = 1.0 - mirrored[:, 0::3]   # Flip X
    # Swap left & right hand
    left = mirrored[:, 99:162].copy()
    right = mirrored[:, 162:225].copy()
    mirrored[:, 99:162] = right
    mirrored[:, 162:225] = left
    return mirrored

def spatial_translation(landmarks, max_shift=0.05):
    """Random small spatial shift (X, Y)"""
    shift_x = np.random.uniform(-max_shift, max_shift)
    shift_y = np.random.uniform(-max_shift, max_shift)
    translated = landmarks.copy()
    translated[:, 0::3] += shift_x   # x coords
    translated[:, 1::3] += shift_y   # y coords
    return translated

def create_mirrored_if_missing(file_path):
    base, ext = os.path.splitext(file_path)
    mirrored_path = f"{base}_mirrored{ext}"
    if not os.path.exists(mirrored_path):
        try:
            landmarks = np.load(file_path)
            mirrored = mirror_augmentation(landmarks)
            np.save(mirrored_path, mirrored)
        except Exception:
            return file_path
    return mirrored_path

# Prepare augmented training list
train_paths = []
train_labels = []
for _, row in train_df.iterrows():
    path = row['npy_path']
    label = row['label']
    
    train_paths.append(path)
    train_labels.append(label)
    
    # Add mirrored version
    mirrored_path = create_mirrored_if_missing(path)
    train_paths.append(mirrored_path)
    train_labels.append(label)

print(f"Training samples after mirror augmentation: {len(train_paths)}")

# ========================= MEAN / STD =========================
print("Computing mean and std from training data...")
all_landmarks = np.concatenate([np.load(p) for p in train_df['npy_path']], axis=0)
mean = np.mean(all_landmarks, axis=0)
std = np.std(all_landmarks, axis=0)
std = np.where(std == 0, 1.0, std)


np.savez('kinematic_stats.npz', mean=mean, std=std)

# ========================= GENERATOR =========================
class SignDataGenerator(Sequence):
    def __init__(self, file_paths, labels, mean, std, batch_size=BATCH_SIZE, shuffle=True, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_files = [self.file_paths[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]

        X = np.zeros((len(batch_files), MAX_FRAMES, 333), dtype=np.float32)

        for i, file_path in enumerate(batch_files):
            landmarks = np.load(file_path)
            
            # Normalization
            normalized = (landmarks - self.mean) / self.std

            # Augmentations (training only)
            if self.augment:
                if np.random.rand() < 0.5:
                    normalized = spatial_translation(normalized, max_shift=0.05)
                if np.random.rand() < 0.6:
                    noise = np.random.normal(0, 0.02, normalized.shape).astype(np.float32)
                    normalized += noise

            X[i] = normalized

        return X, np.array(batch_labels, dtype=np.int32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ========================= GENERATORS =========================
train_generator = SignDataGenerator(train_paths, train_labels, mean, std, 
                                   batch_size=BATCH_SIZE, shuffle=True, augment=True)

val_generator = SignDataGenerator(val_df['npy_path'].tolist(), val_df['label'].tolist(), 
                                 mean, std, batch_size=BATCH_SIZE, shuffle=False, augment=False)

test_generator = SignDataGenerator(test_df['npy_path'].tolist(), test_df['label'].tolist(), 
                                  mean, std, batch_size=BATCH_SIZE, shuffle=False, augment=False)

# ========================= MODEL =========================
def build_model(input_shape=(MAX_FRAMES, 333)):
    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=-999.0, input_shape=input_shape),  # kept as safety
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=tf.keras.regularizers.l2(0.001))),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))
    ])
    return model

model = build_model()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ========================= TRAINING =========================
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("\nStarting training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=MAX_EPOCHS,
    callbacks=callbacks
)

# ========================= EVALUATION =========================
print("\nEvaluating on test set...")
model.evaluate(test_generator)

y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.array(test_generator.labels)   # Note: test_generator.labels may need .labels if not directly available

reverse_label_map = {v: k for k, v in label_map.items()}
class_labels = [reverse_label_map[i] for i in sorted(reverse_label_map.keys())]

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Val")
axes[1].set_title("Loss")
axes[1].legend()
plt.tight_layout()
plt.savefig("accuracy_loss.png", dpi=150)
plt.show()

plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred)
np.save('confusion_matrix.npy', cm)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

print("✓ Training completed. Plots saved.")