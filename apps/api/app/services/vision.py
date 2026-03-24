from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat
from huggingface_hub import hf_hub_download
import tensorflow as tf

from app.core.config import settings
from app.schemas.contracts import ImageAnalysis, VisionPrediction, VisionQuality


ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
logger = logging.getLogger("dermai.api.vision")


@dataclass
class HeuristicMetrics:
    contrast: float
    sharpness: float
    lesion_coverage: float
    asymmetry: float
    center_darkness: float


def generated_dir() -> Path:
    path = Path(__file__).resolve().parents[3] / "generated" / "vision"
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class VisionService:
    def __init__(self) -> None:
        self._model = None
        self._model_path: str | None = None
        self._model_lock = Lock()

    def validate_upload(self, content_type: str | None, payload: bytes) -> None:
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise ValueError("Unsupported image type. Use JPEG, PNG, or WEBP.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise ValueError("Image exceeds the 8 MB upload limit.")

    def load_image(self, payload: bytes) -> Image.Image:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        image.thumbnail((768, 768))
        return image

    def save_upload(self, image: Image.Image) -> Path:
        output_path = generated_dir() / f"{uuid4()}.png"
        image.save(output_path, format="PNG")
        return output_path

    def _mask_from_image(self, image: Image.Image) -> Image.Image:
        grayscale = ImageOps.grayscale(image)
        enhanced = ImageOps.autocontrast(grayscale)
        threshold = max(int(ImageStat.Stat(enhanced).mean[0] * 0.85), 45)
        mask = enhanced.point(lambda pixel: 255 if pixel < threshold else 0)
        return mask.filter(ImageFilter.GaussianBlur(radius=8))

    def _coverage_ratio(self, mask: Image.Image) -> float:
        histogram = mask.histogram()
        bright_pixels = sum(histogram[200:])
        total_pixels = max(mask.size[0] * mask.size[1], 1)
        return round(bright_pixels / total_pixels, 4)

    def _asymmetry_score(self, image: Image.Image) -> float:
        sample = ImageOps.grayscale(image).resize((96, 96))
        flipped_lr = ImageOps.mirror(sample)
        flipped_tb = ImageOps.flip(sample)
        diff_lr = ImageStat.Stat(ImageChops.difference(sample, flipped_lr)).mean[0] / 255
        diff_tb = ImageStat.Stat(ImageChops.difference(sample, flipped_tb)).mean[0] / 255
        return round((diff_lr + diff_tb) / 2, 4)

    def _sharpness_score(self, image: Image.Image) -> float:
        edges = image.filter(ImageFilter.FIND_EDGES).convert("L")
        return round(ImageStat.Stat(edges).mean[0] / 255, 4)

    def _contrast_score(self, image: Image.Image) -> float:
        grayscale = ImageOps.grayscale(image)
        return round(ImageStat.Stat(grayscale).stddev[0] / 128, 4)

    def _center_darkness(self, image: Image.Image) -> float:
        width, height = image.size
        crop = image.crop((width * 0.2, height * 0.2, width * 0.8, height * 0.8))
        mean = ImageStat.Stat(ImageOps.grayscale(crop)).mean[0] / 255
        return round(1 - mean, 4)

    def analyze_metrics(self, image: Image.Image) -> HeuristicMetrics:
        mask = self._mask_from_image(image)
        return HeuristicMetrics(
            contrast=self._contrast_score(image),
            sharpness=self._sharpness_score(image),
            lesion_coverage=self._coverage_ratio(mask),
            asymmetry=self._asymmetry_score(image),
            center_darkness=self._center_darkness(image),
        )

    def confidence_band(self, confidence: float) -> Literal["low", "medium", "high"]:
        if confidence >= 0.72:
            return "high"
        if confidence >= 0.48:
            return "medium"
        return "low"

    def _load_external_model(self):
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            model_path = hf_hub_download(
                repo_id=settings.vision_model_repo_id,
                filename=settings.vision_model_filename,
            )
            self._model_path = model_path
            self._model = tf.keras.models.load_model(model_path)
            logger.info(
                "Loaded external vision model repo_id=%s filename=%s",
                settings.vision_model_repo_id,
                settings.vision_model_filename,
            )
            return self._model

    def _preprocess_for_external_model(self, image: Image.Image) -> np.ndarray:
        resized = image.resize((224, 224))
        array = np.array(resized, dtype=np.float32)
        array = array[..., ::-1]
        array[..., 0] -= 103.939
        array[..., 1] -= 116.779
        array[..., 2] -= 123.68
        return np.expand_dims(array, axis=0)

    def _classify_with_external_model(
        self,
        image: Image.Image,
    ) -> tuple[str, list[VisionPrediction], float]:
        model = self._load_external_model()
        batch = self._preprocess_for_external_model(image)
        prediction = float(model.predict(batch, verbose=0)[0][0])
        malignant_probability = max(0.0, min(prediction, 1.0))
        benign_probability = 1.0 - malignant_probability

        threshold = settings.vision_model_threshold
        predicted_label = "malignant_pattern" if malignant_probability >= threshold else "benign_pattern"
        predicted_confidence = malignant_probability if predicted_label == "malignant_pattern" else benign_probability

        top_predictions = [
            VisionPrediction(
                label="malignant_pattern",
                confidence=round(malignant_probability, 4),
                rationale="Integrated HAM10000-based binary classifier score for a malignant pattern.",
            ),
            VisionPrediction(
                label="benign_pattern",
                confidence=round(benign_probability, 4),
                rationale="Integrated HAM10000-based binary classifier score for a benign pattern.",
            ),
        ]
        top_predictions.sort(key=lambda item: item.confidence, reverse=True)
        return predicted_label, top_predictions, round(predicted_confidence, 4)

    def quality(self, metrics: HeuristicMetrics) -> VisionQuality:
        issues: list[str] = []
        if metrics.sharpness < 0.09:
            issues.append("The image appears soft or slightly blurred.")
        if metrics.contrast < 0.12:
            issues.append("The lesion contrast is limited, which reduces reliability.")
        if metrics.lesion_coverage < 0.03:
            issues.append("The lesion occupies a small portion of the frame.")

        return VisionQuality(
            usable=len(issues) < 3,
            issues=issues,
            contrast=metrics.contrast,
            sharpness=metrics.sharpness,
            lesionCoverage=metrics.lesion_coverage,
            asymmetry=metrics.asymmetry,
        )

    def overlay(self, image: Image.Image) -> Image.Image:
        mask = self._mask_from_image(image).convert("L")
        red_layer = Image.new("RGBA", image.size, (214, 67, 43, 0))
        alpha_mask = mask.point(lambda pixel: min(int(pixel * 0.8), 180))
        red_layer.putalpha(alpha_mask)
        composite = Image.alpha_composite(image.convert("RGBA"), red_layer)
        return composite

    def summarize(self, label: str, quality: VisionQuality) -> tuple[str, str]:
        label_summaries = {
            "malignant_pattern": "The integrated classifier scored this image closer to a malignant-pattern class than a benign one.",
            "benign_pattern": "The integrated classifier scored this image closer to a benign-pattern class than a malignant one.",
        }
        caution = "This image result is best used as visual context for the chat."
        if not quality.usable:
            caution += " Image quality issues can reduce reliability."
        return label_summaries[label], caution

    def analyze(self, payload: bytes, content_type: str | None) -> ImageAnalysis:
        self.validate_upload(content_type, payload)
        image = self.load_image(payload)
        self.save_upload(image)

        metrics = self.analyze_metrics(image)
        label, top_predictions, confidence = self._classify_with_external_model(image)
        quality = self.quality(metrics)
        summary, caution = self.summarize(label, quality)
        overlay = self.overlay(image)

        return ImageAnalysis(
            predictedClass=label,
            confidence=round(confidence, 4),
            confidenceBand=self.confidence_band(confidence),
            summary=summary,
            caution=caution,
            topPredictions=top_predictions,
            quality=quality,
            overlayImageDataUrl=image_to_data_url(overlay),
            width=image.size[0],
            height=image.size[1],
        )


vision_service = VisionService()
