import json
import os
import queue
import threading
import numpy as np
import logging

# Suppress TensorFlow C++ startup outputs to clean up log files
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize the base module logger
base_logger = logging.getLogger(__name__)

class Model_Inference:
    def __init__(self, sequence_queue: queue.Queue, model_path: str, label_map_path: str, on_prediction=None, session_id: str = "SYSTEM"):
        """
        Initializes the Model Inference consumer with multi-user tracking.
        
        :param sequence_queue: The shared queue holding (140, 333) numpy arrays.
        :param model_path: Path to the saved Keras (.h5 or .keras) model.
        :param label_map_path: Path to the JSON file containing the label map.
        :param on_prediction: Optional callback function injected to receive (label, confidence).
        :param session_id: Unique identifier for the user session (e.g., WebSocket channel name).
        """
        self.sequence_queue = sequence_queue
        with open(label_map_path, 'r') as f:
          label_map = json.load(f)
        self.label_map = {v: k for k, v in label_map.items()}
        self.model_path = model_path
        self.on_prediction = on_prediction
        self.model = None
        
        # Wrap the base logger with a contextual adapter for multi-user trace clarity
        self.logger = logging.LoggerAdapter(base_logger, {"session_id": session_id})
        
        self.stop_event = threading.Event()
        self.thread = None

    def load_model(self):
        """Fetches the globally shared single instance of the model via thread-safe lazy loading."""
        # Swapped to use your custom thread-safe shared model singleton
        from pipeline.model_loader import get_model  # type: ignore
        
        self.logger.info("Requesting model instance from global singleton loader...")
        self.model = get_model(self.model_path)
        self.logger.info("Model reference successfully linked to inference worker instance.")

    def start(self):
        """Starts the background consumer inference thread."""
        if self.model is None:
            self.load_model()
            
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True, name=f"INF_{self.logger.extra['session_id']}")
        self.thread.start()
        self.logger.info("Inference consumer execution thread spawned successfully.")

    def stop(self):
        """Gracefully stops the worker thread loop."""
        self.stop_event.set()
        
        # Inject the shutdown sentinel token to unblock queue reading stalls
        self.sequence_queue.put(None) 
        if self.thread:
            self.thread.join()
        self.logger.info("Inference worker thread completely stopped.")

    def _worker_loop(self):
        """The core consumer loop that blocks on q.get() and processes predictions."""
        while not self.stop_event.is_set():
            try:
                sequence = self.sequence_queue.get(timeout=1.0)
                if sequence is None:
                    break

                # 1. Shape transforms from (140, 333) -> (1, 140, 333)
                input_data = np.expand_dims(sequence, axis=0)
                
                # 2. Extract index [0] to safely flatten the (1, 26) batch down to a clean (26,) 1D array
                prediction_probabilities = self.model.predict(input_data, verbose=0)[0]
                
                # 3. Safely calculate integer position and get the individual confidence float
                predicted_class_id = np.argmax(prediction_probabilities)
                confidence = prediction_probabilities[predicted_class_id]
                predicted_label = self.label_map[predicted_class_id]
                
                self.on_prediction_match(predicted_label, float(confidence))
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Critical failure during evaluation: {e}", exc_info=True)

    def on_prediction_match(self, label: str, confidence: float):
        """Dispatches data forward using the injected handler with smart parameter fallback."""
        if self.on_prediction:
            try:
                # Primary Attempt: Try to pass just the essential result variables
                self.on_prediction(label, confidence)
            except TypeError:
                try:
                    # Fallback Attempt: If it expects the session context metadata too
                    session_id = self.logger.extra.get("session_id", "SYSTEM")
                    self.on_prediction(label, confidence, session_id)
                except Exception as e:
                    self.logger.error(f"Failed executing injected on_prediction handler: {e}")
        else:
            self.logger.info(f"Prediction found: {label} | Confidence: {confidence:.2%}")
