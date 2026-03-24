from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass
from typing import Literal

import httpx
from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

from app.core.config import settings
from app.schemas.contracts import ImageAnalysis, VisionPrediction, VisionQuality


ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
logger = logging.getLogger("dermai.api.vision")

HF_LABEL_MAP = {
    "akiec": "actinic_keratosis",
    "bcc": "basal_cell_carcinoma",
    "bkl": "benign_keratosis",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic_nevus",
    "vasc": "vascular_lesion",
}

LABEL_SUMMARIES = {
    "actinic_keratosis": "The model scored this image closest to an actinic-keratosis pattern.",
    "basal_cell_carcinoma": "The model scored this image closest to a basal-cell-carcinoma pattern.",
    "benign_keratosis": "The model scored this image closest to a benign-keratosis pattern.",
    "dermatofibroma": "The model scored this image closest to a dermatofibroma pattern.",
    "melanoma": "The model scored this image closest to a melanoma pattern.",
    "melanocytic_nevus": "The model scored this image closest to a melanocytic-nevus pattern.",
    "vascular_lesion": "The model scored this image closest to a vascular-lesion pattern.",
}

LABEL_RATIONALES = {
    "actinic_keratosis": "Hosted skin-lesion classifier probability for an actinic-keratosis pattern.",
    "basal_cell_carcinoma": "Hosted skin-lesion classifier probability for a basal-cell-carcinoma pattern.",
    "benign_keratosis": "Hosted skin-lesion classifier probability for a benign-keratosis pattern.",
    "dermatofibroma": "Hosted skin-lesion classifier probability for a dermatofibroma pattern.",
    "melanoma": "Hosted skin-lesion classifier probability for a melanoma pattern.",
    "melanocytic_nevus": "Hosted skin-lesion classifier probability for a melanocytic-nevus pattern.",
    "vascular_lesion": "Hosted skin-lesion classifier probability for a vascular-lesion pattern.",
}


@dataclass
class HeuristicMetrics:
    contrast: float
    sharpness: float
    lesion_coverage: float
    asymmetry: float
    center_darkness: float


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class VisionService:
    def validate_upload(self, content_type: str | None, payload: bytes) -> None:
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise ValueError("Unsupported image type. Use JPEG, PNG, or WEBP.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise ValueError("Image exceeds the 8 MB upload limit.")

    def load_image(self, payload: bytes) -> Image.Image:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        image.thumbnail((768, 768))
        return image

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

    def _normalize_label(self, label: str) -> str:
        compact = label.strip().lower()
        if compact in HF_LABEL_MAP:
            return HF_LABEL_MAP[compact]
        compact = re.sub(r"[^a-z0-9]+", "_", compact).strip("_")
        return compact or "unknown_lesion_pattern"

    def _classify_with_hosted_model(
        self,
        payload: bytes,
        content_type: str | None,
    ) -> tuple[str, list[VisionPrediction], float]:
        if not settings.vision_api_key:
            raise RuntimeError("Vision API key is not configured.")

        headers = {
            "Authorization": f"Bearer {settings.vision_api_key}",
        }
        files = {
            "inputs": ("upload-image", payload, content_type or "application/octet-stream"),
        }

        with httpx.Client(timeout=settings.vision_timeout_seconds) as client:
            response = client.post(
                f"{settings.vision_api_base_url.rstrip('/')}/{settings.vision_model_id}",
                headers=headers,
                files=files,
            )

        if response.status_code in {401, 403}:
            raise RuntimeError("Hosted vision authentication failed.")
        if response.status_code == 503:
            raise RuntimeError("Hosted vision model is currently unavailable.")

        response.raise_for_status()
        payload_json = response.json()
        if not isinstance(payload_json, list) or not payload_json:
            raise RuntimeError("Hosted vision model returned an unexpected response.")

        top_predictions: list[VisionPrediction] = []
        for item in payload_json[:5]:
            raw_label = str(item.get("label", "")).strip()
            normalized_label = self._normalize_label(raw_label)
            confidence = round(float(item.get("score", 0.0)), 4)
            top_predictions.append(
                VisionPrediction(
                    label=normalized_label,
                    confidence=confidence,
                    rationale=LABEL_RATIONALES.get(
                        normalized_label,
                        "Hosted skin-lesion classifier probability for this lesion pattern.",
                    ),
                )
            )

        predicted_label = top_predictions[0].label
        predicted_confidence = top_predictions[0].confidence
        return predicted_label, top_predictions, predicted_confidence

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
        summary = LABEL_SUMMARIES.get(label, f"The model scored this image closest to a {label.replace('_', ' ')} pattern.")
        caution = "This image result is best used as visual context for the chat."
        if not quality.usable:
            caution += " Image quality issues can reduce reliability."
        return summary, caution

    def analyze(self, payload: bytes, content_type: str | None) -> ImageAnalysis:
        self.validate_upload(content_type, payload)
        image = self.load_image(payload)

        metrics = self.analyze_metrics(image)
        label, top_predictions, confidence = self._classify_with_hosted_model(payload, content_type)
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
