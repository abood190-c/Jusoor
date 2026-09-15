import asyncio
import json
import logging
import queue

from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings

# Import the optimized, single-import package storefront
from .pipeline import Frame_To_Sequence, Model_Inference, State

logger = logging.getLogger(__name__)

class SignLanguageConsumer(AsyncWebsocketConsumer):
    
    async def connect(self):
        """Triggered on initial WebSocket handshake request."""
        # 1. Capture the running ASGI event loop context immediately
        self.async_loop = asyncio.get_running_loop()
        
        # 2. Derive a unique session tracking ID from the Channel name
        self.session_id = self.channel_name.split('.')[-1]
        logger.info(f"Incoming connection handshake. Initializing session: {self.session_id}")

        # 3. Accept the WebSocket connection handshake 
        await self.accept()

        try:
            # 4. Retrieve model paths and classification label maps from settings.py
            model_path = getattr(settings, 'SIGN_LANGUAGE_MODEL_PATH', 'models/action_recognition.keras')
            label_map_path = getattr(settings, 'SIGN_LANGUAGE_LABEL_MAP_PATH', 'models/label_map.json')

            # 5. Initialize communication resources with real-time zero-lag constraints (maxsize=1)
            self.sequence_queue = queue.Queue(maxsize=1)
            
            # Defer creation and management of Frame_Capturing to the underlying classes
            # pipeline/__init__.py storefront instantiates it contextually
            from .pipeline.frame_capturing import Frame_Capturing
            self.capturing = Frame_Capturing(session_id=self.session_id)

            stats_path = getattr(settings, 'SIGN_LANGUAGE_STATS_PATH', 'models/kinematic_stats.npz')
            model_path = getattr(settings, 'SIGN_LANGUAGE_MODEL_PATH', 'models/holistic_landmarker.task')
            self.frame_to_sequence = Frame_To_Sequence(
                sequence_queue=self.sequence_queue,
                capturing=self.capturing,
                session_id=self.session_id,
                stats_path=stats_path,
                model_path=model_path
            )

            # 6. Define the nested threadsafe prediction callback handler
            def on_prediction_received(label: str, confidence: float):
                """Callback handler executed directly by the background Model_Inference thread."""
                payload = {
                    "type": "gesture_result",
                    "prediction": label,
                    "confidence": round(confidence, 4)
                }
                # Fire-and-forget: dispatch the coroutine onto the ASGI event loop from Thread 3
                asyncio.run_coroutine_threadsafe(
                    self.send(text_data=json.dumps(payload)), 
                    self.async_loop
                )

            # 7. Instantiate the Inference engine with centralized callback injection
            self.model_inference = Model_Inference(
                sequence_queue=self.sequence_queue,
                model_path=model_path,
                label_map_path=label_map_path,
                on_prediction=on_prediction_received,
                session_id=self.session_id
            )

            # 8. Start background threads (frame_to_sequence internally handles capturing.start())
            self.frame_to_sequence.start()
            self.model_inference.start()

            logger.info(f"All pipeline threads running successfully for session: {self.session_id}")

        except Exception as e:
            # Safely log using the fallback string extraction token
            sid = getattr(self, 'session_id', 'UNKNOWN')
            logger.error(f"Critical initialization failure for session {sid}: {e}", exc_info=True)
            
            # Inform frontend client UI components of the error before shutting down the pipeline
            try:
                await self.send(text_data=json.dumps({"type": "error", "message": "Pipeline initialization failed."}))
            except Exception:
                pass
            await self.close()

    async def receive(self, text_data=None, bytes_data=None):
        """Triggered on websocket_receive. Routes binary video data payloads."""
        # RESOLVED EDGE CASE 2: Safe frame handling guard verification check
        if bytes_data and hasattr(self, 'capturing') and self.capturing:
            self.capturing.push_frame(bytes_data)

    async def disconnect(self, close_code):
        """Triggered on websocket_disconnect. Cleans up background thread lifecycles."""
        # RESOLVED EDGE CASE 1: Fallback extraction parsing protection logic
        sid = getattr(self, 'session_id', 'UNKNOWN')
        logger.info(f"WebSocket closed (Code: {close_code}). Cleaning up session: {sid}")
        
        # Guard references ensure zombie threads are never left running in memory space
        if hasattr(self, 'frame_to_sequence') and self.frame_to_sequence:
            try:
                self.frame_to_sequence.stop()
            except Exception as e:
                logger.error(f"Error stopping frame_to_sequence for session {sid}: {e}")
            
        if hasattr(self, 'model_inference') and self.model_inference:
            try:
                self.model_inference.stop()
            except Exception as e:
                logger.error(f"Error stopping model_inference for session {sid}: {e}")

        logger.info(f"Pipeline threads completely terminated for session: {sid}")
