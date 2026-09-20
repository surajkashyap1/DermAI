"""Load the trained CNN and run predictions with Grad-CAM overlays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from dermai.classifier.gradcam import compute_heatmap, overlay_heatmap
from dermai.classifier.labels import ClassInfo, class_info
from dermai.classifier.model import build_model
from dermai.config import IMAGE_SIZE, MODEL_PATH


@dataclass
class Prediction:
    class_info: ClassInfo
    confidence: float                    # 0..1 for the top class
    probabilities: list[float]           # per-class, index-aligned to CLASSES
    overlay: Image.Image | None          # Grad-CAM overlay on the original image

    @property
    def label(self) -> str:
        return self.class_info.name


def preprocess(image: Image.Image, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize to the model's input and shape as a single-item batch."""
    rgb = image.convert("RGB").resize((image_size, image_size))
    array = np.asarray(rgb, dtype=np.float32)
    return array.reshape(1, image_size, image_size, 3)


class SkinLesionClassifier:
    """Lazily loads ``best_model.h5`` and serves predictions + Grad-CAM."""

    def __init__(self, model_path: Path = MODEL_PATH, image_size: int = IMAGE_SIZE):
        self.model_path = Path(model_path)
        self.image_size = image_size
        self._model = None

    @property
    def is_available(self) -> bool:
        return self.model_path.exists()

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.is_available:
            raise FileNotFoundError(
                f"Model weights not found at {self.model_path}. Train the CNN with "
                "`python -m dermai.classifier.train` or place best_model.h5 in models/."
            )
        # Rebuild the graph and load weights (robust across Keras versions).
        model = build_model(self.image_size)
        model.load_weights(str(self.model_path))
        self._model = model
        return model

    def predict(self, image: Image.Image, with_gradcam: bool = True) -> Prediction:
        model = self._load()
        batch = preprocess(image, self.image_size)
        probs = model.predict(batch, verbose=0)[0]
        top_index = int(np.argmax(probs))

        overlay = None
        if with_gradcam:
            try:
                heatmap = compute_heatmap(model, batch, class_index=top_index)
                overlay = overlay_heatmap(image, heatmap)
            except Exception:
                # Grad-CAM is best-effort; never block a prediction on it.
                overlay = None

        return Prediction(
            class_info=class_info(top_index),
            confidence=float(probs[top_index]),
            probabilities=[float(p) for p in probs],
            overlay=overlay,
        )
