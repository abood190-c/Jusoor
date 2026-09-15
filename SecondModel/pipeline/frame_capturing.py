import numpy as np
import queue
import threading
import mediapipe as mp
import cv2


class Frame_Capturing:
    def __init__(self):
        """Initializes the memory-buffered frame manager for network video streams."""
        self.lock = threading.Lock()
        self.latest_frame = None
        
        # Kept to preserve pipeline lifecycle compatibility with external start/stop calls
        self.is_running = False

    def start(self):
        """Prepares the component to receive incoming memory buffers."""
        with self.lock:
            self.latest_frame = None
            self.is_running = True

    def stop(self):
        """Clears buffers and marks the capture status as inactive."""
        with self.lock:
            self.latest_frame = None
            self.is_running = False

    def push_frame(self, encoded_bytes: bytes):
        """
        Receives binary camera data directly from the WebSocket connection loop,
        decodes it via OpenCV, and updates the shared memory buffer.
        
        :param encoded_bytes: Raw binary image payload (e.g., JPEG or PNG data string).
        """
        if not encoded_bytes:
            return

        np_array = np.frombuffer(encoded_bytes, dtype=np.uint8)
        
        decoded_frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Guard against corrupted or undecodable image payloads
        if decoded_frame is None:
            # Silently drop the broken chunk and retain the last working frame
            return

        with self.lock:
            if self.is_running:
                self.latest_frame = decoded_frame

    def get_frame(self):
        """
        Fetches the latest decoded frame from memory.
        
        :return: A safe deep-copy matrix of the frame if present, otherwise None.
        """
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None
