"""HAM10000 7-class skin-lesion CNN with Grad-CAM explainability."""

from dermai.classifier.labels import CLASS_LABELS, CLASSES, class_info
from dermai.classifier.inference import SkinLesionClassifier, Prediction

__all__ = [
    "CLASS_LABELS",
    "CLASSES",
    "class_info",
    "SkinLesionClassifier",
    "Prediction",
]
