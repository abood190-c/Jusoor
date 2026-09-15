# Sign Language Pipeline — Backend Integration Notes

## Files Provided
- `pipeline/` — core AI pipeline package
- `consumers.py` — Django Channels WebSocket consumer
- `model.keras` — trained Bi-LSTM word-level ASL model
- `label_map.json` — 26-class label index reference
- `requirements.txt` — exact dependency versions

## Django Settings Required

### 1. INSTALLED_APPS
```python
INSTALLED_APPS = [
    ...
    'channels',
    'daphne',
]
```

### 2. ASGI Application
```python
ASGI_APPLICATION = 'your_project_name.asgi.application'
```

### 3. Channel Layers (demo)
```python
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels.layers.InMemoryChannelLayer'
    }
}
```

### 4. Model Configuration
```python
SIGN_LANGUAGE_MODEL_PATH = 'path/to/model.keras'
SIGN_LANGUAGE_LABEL_MAP = [
    "accident", "basketball", "bed", "before", "bowling",
    "call", "candy", "change", "cold", "computer",
    "cool", "corn", "cousin", "dark", "drink",
    "go", "help", "last", "later", "pizza",
    "shirt", "short", "tall", "thin", "trade", "who"
]
```

### 5. Logging
```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'pipeline': {
            'format': '[%(asctime)s] [%(session_id)s] %(levelname)s %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'pipeline',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG',
    },
}
```

## Routing
```python
# routing.py
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/sign-language/$', 
            consumers.SignLanguageConsumer.as_asgi()),
]
```

## WebSocket Protocol Contract
- Client sends: **binary frames** (JPEG bytes) — not base64 text
- Server responds: JSON text
```json
{"type": "gesture_result", "prediction": "hello", "confidence": 0.94}
{"type": "error", "message": "Pipeline initialization failed."}
```

## Critical Notes
- Run with **Daphne**, not `manage.py runserver`
- Python **3.11** required — do not use 3.12+
- Each WebSocket connection spawns its own pipeline instance
- Model loads once globally via singleton — thread-safe