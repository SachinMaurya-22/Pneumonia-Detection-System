
"""TensorFlow/Keras inference service."""

import json
import os
from pathlib import Path
from threading import Lock

import numpy as np
import tensorflow as tf

from app.utils import preprocess_image


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        str(BASE_DIR / "models" / "pneumonia_model.keras"),
    )
)

CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

# These labels must match the training class order.
DEFAULT_CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

_model = None
_model_lock = Lock()


def load_class_names():
    """Load class labels or use the documented defaults."""

    if CLASS_NAMES_PATH.exists():
        with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
            labels = json.load(file)

        if (
            not isinstance(labels, list)
            or not labels
            or not all(isinstance(x, str) for x in labels)
        ):
            raise ValueError("Invalid class_names.json format.")

        return labels

    return DEFAULT_CLASS_NAMES


def get_model():
    """Load the trained model once, on first use."""

    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                if not MODEL_PATH.is_file():
                    raise FileNotFoundError(
                        f"Trained model not found: {MODEL_PATH}"
                    )

                _model = tf.keras.models.load_model(
                    MODEL_PATH,
                    compile=False,
                )

    return _model


def predict_image(file_bytes: bytes) -> dict:
    """
    Predict a class for one image.

    Supported output assumptions:
      - Binary sigmoid: shape (1, 1)
      - Multiclass probabilities: shape (1, number_of_classes)

    Returns a label and model score.
    """

    model = get_model()
    class_names = load_class_names()

    # Match image dimensions to the model input where possible.
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        raise ValueError("Multiple-input models are not supported.")

    height = input_shape[1] or 224
    width = input_shape[2] or 224

    batch = preprocess_image(
        file_bytes,
        image_size=(int(width), int(height)),
    )

    raw_output = np.asarray(
        model.predict(batch, verbose=0),
        dtype=np.float64,
    )

    if raw_output.ndim != 2 or raw_output.shape[0] != 1:
        raise ValueError("Unexpected model output shape.")

    scores = raw_output[0]

    if not np.all(np.isfinite(scores)):
        raise ValueError("Model returned invalid numerical values.")

    # Binary sigmoid output.
    if scores.size == 1 and len(class_names) == 2:
        positive_score = float(scores[0])

        if not 0.0 <= positive_score <= 1.0:
            raise ValueError(
                "Binary output must be a sigmoid probability."
            )

        probabilities = np.array([
            1.0 - positive_score,
            positive_score,
        ])

    # Multiclass softmax output.
    elif scores.size == len(class_names) and scores.size >= 2:
        if (
            np.any(scores < 0)
            or np.any(scores > 1)
            or not np.isclose(scores.sum(), 1.0, atol=1e-3)
        ):
            raise ValueError(
                "Expected multiclass softmax probabilities. "
                "Logits require the correct training-specific conversion."
            )

        probabilities = scores / scores.sum()

    else:
        raise ValueError(
            "Model output dimensions do not match class labels."
        )

    index = int(np.argmax(probabilities))

    return {
        "prediction": class_names[index],
        "confidence": float(probabilities[index] * 100.0),
    }
