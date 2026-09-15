import tempfile
import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
import queue
import threading
import logging
import time  # Added to prevent busy-loop CPU starvation

from pipeline.frame_capturing import Frame_Capturing
from pipeline.utils import State



from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode


logger = logging.getLogger(__name__)

class Frame_To_Sequence:
    def __init__(self, sequence_queue: queue.Queue, capturing: Frame_Capturing, session_id: str = "SYSTEM",stats_path: str = "kinematic_stats.npz",model_path: str = "holistic_landmarker.task"):
        """
        Initializes the MediaPipe state machine execution worker thread.
        
        :param sequence_queue: Bounded tracking buffer queue shared with inference engine.
        :param capturing: Target frame capturing frame buffer proxy.
        :param session_id: String session context tracking ID injected from consumer.
        """
        self.sequence_queue = sequence_queue
        self.capturing = capturing
        self.state = State.NEUTRAL
        self.target_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        self.model_url = ("https://storage.googleapis.com/mediapipe-models/"
             "holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task")
        self.model_path = model_path

        self.recording_buffer = []

        self.logger = logging.LoggerAdapter(logger, {"session_id": session_id})


            # Initialize Modern Tasks API
        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.VIDEO,
            output_face_blendshapes=False,
        )
        self.holistic_landmarker = HolisticLandmarker.create_from_options(options)

        self.stop_event = threading.Event()
        self.thread = None
        self.prev_left_wrist = None
        self.prev_right_wrist = None
        self.frame_count = 0

        
        # Load Offline Training Statistics for Normalization
        try:
            stats = np.load(stats_path)
            self.mean = stats['mean']      # Expected shape: (333,)
            self.std = stats['std']        # Expected shape: (333,)
            # Prevent Division by Zero on constant features
            self.std[self.std == 0.0] = 1e-6
        except Exception as e:
            logger.error(f"Failed to load normalization statistics from {stats_path}: {e}")
            raise e


    def download_model_if_needed(self, model_file, url):
        """Checks for the presence of the MediaPipe holistic landmarker model file and downloads it if missing."""
        if os.path.exists(model_file):
            print(f"✓ Found model file: {model_file}")
            return
        print(f"Downloading MediaPipe holistic landmarker model ...")
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
            print(f"❌ Download failed! Cleaned up partial file. Error: {e}")
            raise e

    def start(self):
        """Launches background tracking worker processor loops."""
        self.download_model_if_needed(self.model_path, self.model_url)
        self.capturing.start()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.update, daemon=True, name=f"F2S_{self.logger.extra['session_id']}")
        self.thread.start()
        self.logger.info("Tracking sequence processor thread spawned successfully.")

    def stop(self):
        """Gracefully terminates worker thread tracking components and unloads graphs."""
        self.stop_event.set()
        if self.thread:
            self.thread.join()
            self.capturing.stop()
        
        if self.holistic_landmarker:
            self.holistic_landmarker.close()
            self.logger.info("MediaPipe Holistic graph engine shut down cleanly.")

    def landmark_coord(self, landmarks, ID):
        """Safely extracts spatial point vectors with NaN structural protections."""
        if landmarks is None or len(landmarks) < (ID * 3 + 3):
            return np.array([0.0, 0.0, 0.0])
        
        coord = landmarks[ID * 3 : ID * 3 + 3]
        # If the coordinates are NaN, return a clean zero vector or handle safely
        if np.isnan(coord).any():
            return np.array([0.0, 0.0, 0.0])
            
        return coord

    def update(self):
        """Main evaluation loop executing continuously over network frames."""
        while not self.stop_event.is_set():
            frame = self.capturing.get_frame()
            if frame is not None:
                self.frame_count += 1
                landmarks = self.extract_landmarks_holistic(frame)
                
                # Check if the core pose segment (first 99 elements) contains valid data
                has_pose = landmarks is not None and not np.isnan(landmarks[:99]).any()
                
                if has_pose:
                    is_neutral = self.is_neutral_position(landmarks)
                    left_wrist_velocity = self.get_velocity(self.landmark_coord(landmarks, 15), wrist_side='left')
                    right_wrist_velocity = self.get_velocity(self.landmark_coord(landmarks, 16), wrist_side='right')
                else:
                    is_neutral = False
                    left_wrist_velocity = 0.0
                    right_wrist_velocity = 0.0
                    # Optional: reset wrist history during total tracking blackout to prevent teleport jumps
                    self.prev_left_wrist = None
                    self.prev_right_wrist = None

                self._run_state_machine(landmarks, is_neutral, left_wrist_velocity, right_wrist_velocity)
            else:
                self.logger.debug("No camera frame available. Throttling thread cycles.")
                time.sleep(0.01)


    def extract_landmarks_holistic(self, frame):
        """Processes RGB matrix structures to output a consistent flattened 333 feature spatial vector using modern Tasks API."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert OpenCV image to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate timestamp in milliseconds
        timestamp_ms = int((self.frame_count / 30.0) * 1000)
        
        # Modern Tasks API call for VIDEO mode
        results = self.holistic_landmarker.detect_for_video(mp_image, timestamp_ms)

        
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
                                        for i in self.target_indices]).flatten()
        else:
            face_landmarks = np.full(len(self.target_indices) * 3, np.nan)

        return np.concatenate([pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks])


    def is_neutral_position(self, landmarks):
        """Evaluates absolute joint spatial metrics to isolate rest position postures."""
        if landmarks is None:
            return True
            
        left_wrist = self.landmark_coord(landmarks, 15)
        right_wrist = self.landmark_coord(landmarks, 16)
        
        # If tracking was lost on the wrists, we can't reliably confirm neutrality
        if np.all(left_wrist == 0.0) or np.all(right_wrist == 0.0):
            return False  
        
        left_hip = self.landmark_coord(landmarks, 23)
        right_hip = self.landmark_coord(landmarks, 24)
        
        left_wrist_to_hip = np.linalg.norm(left_wrist - left_hip)
        right_wrist_to_hip = np.linalg.norm(right_wrist - right_hip)
        
        return left_wrist_to_hip < 0.1 and right_wrist_to_hip < 0.1

    def get_velocity(self, current_wrist_position, wrist_side='left'):
        """Calculates historical displacement delta vectors across running updates."""
        if wrist_side == 'left':
            if self.prev_left_wrist is None:
                self.prev_left_wrist = current_wrist_position
                return 0.0
            distance = np.linalg.norm(current_wrist_position - self.prev_left_wrist)
            self.prev_left_wrist = current_wrist_position
        else:
            if self.prev_right_wrist is None:
                self.prev_right_wrist = current_wrist_position
                return 0.0
            distance = np.linalg.norm(current_wrist_position - self.prev_right_wrist)
            self.prev_right_wrist = current_wrist_position

        return distance / 0.033  # Approximation for ~30 FPS frame targets

    def _run_state_machine(self, landmarks, is_neutral, left_wrist_velocity, right_wrist_velocity):
        """Drives state changes based on spatial thresholds and velocities."""
        if self.state == State.NEUTRAL:
            if not is_neutral and (left_wrist_velocity > 0.5 or right_wrist_velocity > 0.5):
                self.state = State.CHECKING
                self.logger.info("Motion detected! Transitioning: NEUTRAL -> CHECKING")
                
        elif self.state == State.CHECKING:
            if not is_neutral and (left_wrist_velocity > 0.5 or right_wrist_velocity > 0.5):
                self.state = State.RECORDING
                self.logger.info("Gesture verified! Transitioning: CHECKING -> RECORDING")
                if landmarks is not None:
                    self.recording_buffer.append(landmarks)
            else:
                self.state = State.NEUTRAL
                self.logger.debug("False alarm motion signature. Reverting to NEUTRAL.")

        elif self.state == State.RECORDING:
            has_valid_pose = landmarks is not None and not np.isnan(landmarks[:99]).any()

            if has_valid_pose:
                self.recording_buffer.append(landmarks)
            else:
                self.logger.warning("Tracking lost during recording! Appending NaN frame to buffer.")
                self.recording_buffer.append(np.full(333, np.nan))

            if is_neutral or (left_wrist_velocity < 0.5 and right_wrist_velocity < 0.5) or len(self.recording_buffer) >= 140:
                if len(self.recording_buffer) >= 25:
                    self.logger.info(f"Recording complete. Captured {len(self.recording_buffer)} frames. Transitioning -> SENDING")
                    self._handle_sending_state()
                else:
                    self.logger.warning(f"Discarding segment: dropped below 25 frame limit ({len(self.recording_buffer)} frames).")
                    self.recording_buffer.clear()
                    self.state = State.NEUTRAL

    def _handle_sending_state(self):
        """Processes raw frames via Imputation, Resampling, and Global Normalization before dispatching."""
        
        sequence = np.array(self.recording_buffer)
        T, F = sequence.shape

         
        # PHASE 1: LINEAR IMPUTATION WITH NEAREST-NEIGHBOR EDGE FILL
        
        # Interpolate missing tracking joints (NaNs) column by column
        for col in range(F):
            y = sequence[:, col]
            nans = np.isnan(y)
            if not np.any(nans):
                continue
            
            # Find indices of valid numbers
            valid_idx = np.where(~nans)[0]
            
            if len(valid_idx) == 0:
                # SCENARIO A: Total tracking blackout for this feature across the whole sequence.
                # Since we have zero frame reference, a zero-fill fallback is required.
                sequence[:, col] = 0.0
            else:
                # SCENARIO B: Internal gaps or edge gaps exist.
                nan_idx = np.where(nans)[0]
                
                # np.interp automatically forward-fills and backward-fills edge NaNs 
                # using the nearest valid values if left/right are not specified.
                sequence[nan_idx, col] = np.interp(
                    nan_idx, 
                    valid_idx, 
                    y[valid_idx]
                )
         
        # PHASE 2: TEMPORAL RESAMPLING (Target exactly 140 frames)
         
        TARGET_LEN = 140
        if T != TARGET_LEN:
            # Construct original and target time grids
            original_timeline = np.linspace(0, 1, num=T)
            target_timeline = np.linspace(0, 1, num=TARGET_LEN)
            
            resampled_sequence = np.zeros((TARGET_LEN, F))
            for col in range(F):
                resampled_sequence[:, col] = np.interp(target_timeline, original_timeline, sequence[:, col])
            sequence = resampled_sequence

         
        # PHASE 3: GLOBAL FEATURE NORMALIZATION (Z-Score)
         
        # Broadcast subtraction of offline mean and division of offline std across all 140 frames
        normalized_sequence = (sequence - self.mean) / self.std

         
        # PHASE 4: BOUNDED DISPATCH
         
        try:
            self.sequence_queue.put_nowait(normalized_sequence)
            self.logger.info("Normalized sequence successfully pushed to inference engine.")
        except queue.Full:
            try:
                self.sequence_queue.get_nowait()  # Drop oldest out-of-date array
            except queue.Empty:
                pass
            try:
                self.sequence_queue.put_nowait(normalized_sequence)
            except queue.Full:
                self.logger.error("Queue deadlock encountered. Unable to dispatch sequence arrays.")

        self.recording_buffer.clear()
        self.state = State.NEUTRAL
