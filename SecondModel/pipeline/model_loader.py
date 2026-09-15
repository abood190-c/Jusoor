import threading
import logging

logger = logging.getLogger(__name__)

_model = None
_current_path = None
_model_lock = threading.Lock()

def get_model(model_path: str):
    global _model, _current_path
    
    with _model_lock:
        if _model is None or _current_path != model_path:
            from tensorflow.keras.models import load_model # type:ignore
            
            if _model is not None:
                logger.warning(f"[ModelLoader] Path changed from {_current_path} to {model_path}. Overwriting cache.")
            else:
                logger.info(f"[ModelLoader] Initializing model cache for: {model_path}")
            
            _model = load_model(model_path)
            _current_path = model_path

        return _model