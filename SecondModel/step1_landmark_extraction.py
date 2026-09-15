import os
import urllib.request
import tempfile
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
from scipy.interpolate import interp1d
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode

# ─── CONFIGURATION ───
ASL_CITIZEN_DIR = "/kaggle/input/datasets/abd0kamel/asl-citizen/ASL_Citizen/videos/"
FILTERED_SPLITS_DIR = "/kaggle/input/datasets/abdalrhmanashqar/filtered-splits/"
OUTPUT_LANDMARK_DIR = "/kaggle/working/landmark_dir_v2/"
MODEL_PATH = "/kaggle/working/holistic_landmarker.task"

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task")

TARGET_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

# ─── PIPELINE DEPENDENCY SETUP ───
def download_model_if_needed(model_file, url):
    if os.path.exists(model_file):
        print(f"✓ Found model file: {model_file}")
        return
    print(f"Downloading MediaPipe holistic landmarker model (~400MB)...")
    target_dir = os.path.dirname(os.path.abspath(model_file)) or os.getcwd()
    with tempfile.NamedTemporaryFile(dir=target_dir, delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, model_file)
        print(f"✓ Downloaded to {model_file}")
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"❌ Download failed! Error: {e}")
        raise e

download_model_if_needed(MODEL_PATH, MODEL_URL)

# Global options (we recreate the landmarker per video)
options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    output_face_blendshapes=False,
)

# ─── SIGNAL DECOUPLING & TIME NORMALIZATION ───
def impute_and_resample_sequence(raw_matrix, target_frames=140):
    n_frames = raw_matrix.shape[0]
    if n_frames < 2:
        return np.zeros((target_frames, 333), dtype=np.float32)
        
    df = pd.DataFrame(raw_matrix)
    df_imputed = df.ffill().bfill()
    if df_imputed.isnull().values.any():
        df_imputed = df_imputed.fillna(0.5)
        
    imputed_matrix = df_imputed.to_numpy()
    
    original_timeline = np.arange(n_frames)
    target_timeline = np.linspace(0, n_frames - 1, num=target_frames)
    interpolator = interp1d(original_timeline, imputed_matrix, axis=0, kind='linear')
    return interpolator(target_timeline).astype(np.float32)

def validate_processed_matrix(matrix, video_path):
    assert matrix.shape == (140, 333), f"❌ Shape Error for {video_path}: Expected (140, 333), got {matrix.shape}"
    assert matrix.dtype == np.float32, f"❌ Dtype Error for {video_path}: Expected float32, got {matrix.dtype}"
    assert not np.isnan(matrix).any(), f"❌ Leakage Error for {video_path}: NaN values found!"
    variance_per_feature = np.var(matrix, axis=0)
    assert np.max(variance_per_feature) > 1e-5, f"❌ Kinetic Error for {video_path}: Skeleton frozen!"
    assert np.min(matrix) >= -5.0 and np.max(matrix) <= 5.0, (
        f"❌ Boundary Error for {video_path}: Out of bounds! Min: {np.min(matrix)}, Max: {np.max(matrix)}"
    )

# ─── EXTRACTION ───
def extract_asl_citizen_landmarks(split_csv_name):
    csv_path = os.path.join(FILTERED_SPLITS_DIR, split_csv_name)
    if not os.path.exists(csv_path):
        print(f"⚠️ Filtered split not found at: {csv_path}. Skipping.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"\nProcessing {split_csv_name} ({len(df)} videos)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        gloss = str(row['Gloss']).strip()
        video_rel_path = row['Video file']
        video_path = os.path.join(ASL_CITIZEN_DIR, video_rel_path)
        
        if not os.path.exists(video_path):
            continue
            
        video_filename = os.path.basename(video_rel_path)
        npy_filename = video_filename.replace('.mp4', '.npy')
        class_output_dir = os.path.join(OUTPUT_LANDMARK_DIR, gloss)
        os.makedirs(class_output_dir, exist_ok=True)
        
        landmark_save_path = os.path.join(class_output_dir, npy_filename)
        if os.path.exists(landmark_save_path):
            continue

        # === Create fresh landmarker for this video (critical fix) ===
        holistic_landmarker = HolisticLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(video_path)
        raw_clip_landmarks = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Frame-based timestamp starting from 0 for each video
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            timestamp_ms = int((frame_count * 1000) / fps)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            try:
                results = holistic_landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                print(f"MediaPipe error on {video_path} frame {frame_count}: {e}")
                if raw_clip_landmarks:
                    raw_clip_landmarks.append(raw_clip_landmarks[-1].copy())
                else:
                    raw_clip_landmarks.append(np.full(333, np.nan))
                frame_count += 1
                continue

            # POSE
            if results.pose_landmarks:
                pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks]).flatten()
            else:
                pose_landmarks = np.full(33 * 3, np.nan)

            # LEFT HAND
            if results.left_hand_landmarks:
                left_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks]).flatten()
            else:
                left_hand_landmarks = np.full(21 * 3, np.nan)

            # RIGHT HAND
            if results.right_hand_landmarks:
                right_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks]).flatten()
            else:
                right_hand_landmarks = np.full(21 * 3, np.nan)

            # FACE
            if results.face_landmarks:
                face_lms = results.face_landmarks
                face_landmarks = np.array([[face_lms[i].x, face_lms[i].y, face_lms[i].z] 
                                         for i in TARGET_INDICES]).flatten()
            else:
                face_landmarks = np.full(len(TARGET_INDICES) * 3, np.nan)

            landmarks = np.concatenate([pose_landmarks, left_hand_landmarks, 
                                      right_hand_landmarks, face_landmarks])
            raw_clip_landmarks.append(landmarks)
            frame_count += 1

        cap.release()

        if not raw_clip_landmarks:
            print(f"⚠️ No frames extracted for {video_path}")
            continue

        raw_matrix = np.array(raw_clip_landmarks, dtype=np.float32)
        final_processed_matrix = impute_and_resample_sequence(raw_matrix, target_frames=140)
        
        try:
            validate_processed_matrix(final_processed_matrix, video_path)
            np.save(landmark_save_path, final_processed_matrix)
            if idx % 100 == 0:
                print(f"✓ Processed {idx}: {gloss} -> {landmark_save_path}")
        except AssertionError as e:
            print(f"❌ Validation failed for {video_path}: {e}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_LANDMARK_DIR, exist_ok=True)
    for split in ["train.csv", "val.csv", "test.csv"]:
        extract_asl_citizen_landmarks(split)
    print("\n✓ Entire extraction and time normalization process completed successfully.")