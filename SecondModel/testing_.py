import time
import queue
import logging
import os

# Configure basic logging to see pipeline output in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standardize imports assuming you run this script from the parent directory of 'pipeline'
from pipeline import Frame_To_Sequence, Model_Inference
from pipeline.frame_capturing import Frame_Capturing

def dummy_prediction_callback(label: str, confidence: float):
    """
    Replaces the async WebSocket send. This is called directly 
    by the background Model_Inference thread.
    """
    print(f"\n[PREDICTION] Detected Gesture: '{label}' | Confidence: {confidence:.4f}\n")

def main():
    session_id = "local_test_session"
    logger.info(f"Starting standalone local pipeline test for session: {session_id}")

    # 1. Define paths (Adjust these to point to your actual local files)
    model_path = 'sign_lstm_model_v3.keras'
    label_map_path = 'label_map_v3.json'
    stats_path = 'kinematic_stats.npz'
    
    # 2. Replicate the Django Consumer connection initialization setup
    sequence_queue = queue.Queue(maxsize=1)
    capturing = Frame_Capturing(session_id=session_id)
    
    frame_to_sequence = Frame_To_Sequence(
        sequence_queue=sequence_queue,
        capturing=capturing,
        session_id=session_id,
        stats_path=stats_path,
        model_path=model_path
    )
    
    model_inference = Model_Inference(
        sequence_queue=sequence_queue,
        model_path=model_path,
        label_map_path=label_map_path,
        on_prediction=dummy_prediction_callback,  # Injecting our simple print callback
        session_id=session_id
    )

    try:
        # 3. Start background processing threads
        logger.info("Starting pipeline background threads...")
        frame_to_sequence.start()
        model_inference.start()
        
        # 4. Simulate sending streaming video frames
        logger.info("Simulating streaming frame input (sending dummy bytes)...")
        
        # Simulating 50 frames arriving at ~30 FPS
        for i in range(50):
            # Replace b"dummy_frame_bytes" with actual image bytes from a file or opencv if desired
            dummy_frame_bytes = b"0" * 1024  
            
            # Mimic the 'receive' method from your consumer
            capturing.push_frame(dummy_frame_bytes)
            
            # Control feed rate (approx 33ms per frame)
            time.sleep(0.033)
            
        # Give the model a brief window to finish any remaining queue processing
        logger.info("Finished pushing dummy frames. Waiting briefly for lingering inferences...")
        time.sleep(2)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
        
    finally:
        # 5. Replicate the 'disconnect' cleanup workflow to avoid zombie threads
        logger.info("Teardown initiated. Stopping pipeline threads...")
        if frame_to_sequence:
            try:
                frame_to_sequence.stop()
            except Exception as e:
                logger.error(f"Error stopping frame_to_sequence: {e}")
                
        if model_inference:
            try:
                model_inference.stop()
            except Exception as e:
                logger.error(f"Error stopping model_inference: {e}")
                
        logger.info("Local pipeline execution completed successfully.")

if __name__ == "__main__":
    main()