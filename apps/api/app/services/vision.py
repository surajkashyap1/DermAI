from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

from app.schemas.contracts import ImageAnalysis, VisionPrediction, VisionQuality


ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MODEL_NAME = "dermai-vision-demo-v1"


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

    def classify(self, metrics: HeuristicMetrics) -> tuple[str, list[VisionPrediction], float]:
        suspicious = (
            metrics.asymmetry * 0.45
            + metrics.center_darkness * 0.2
            + metrics.lesion_coverage * 0.15
            + metrics.contrast * 0.2
        )
        benign = (1 - metrics.asymmetry) * 0.35 + metrics.sharpness * 0.2 + (1 - metrics.lesion_coverage) * 0.45
        low_quality = max(0.0, 0.7 - metrics.sharpness) * 0.7 + max(0.0, 0.18 - metrics.contrast) * 0.3
        indeterminate = 0.55 + abs(suspicious - benign) * -0.35 + metrics.lesion_coverage * 0.15

        raw_scores = {
            "suspicious_irregular_pattern": max(suspicious, 0.01),
            "uniform_benign_like_pattern": max(benign, 0.01),
            "indeterminate_pigmented_pattern": max(indeterminate, 0.01),
            "low_quality_capture": max(low_quality, 0.01),
        }
        total = sum(raw_scores.values())
        normalized = {label: round(score / total, 4) for label, score in raw_scores.items()}
        ordered = sorted(normalized.items(), key=lambda item: item[1], reverse=True)[:3]

        rationales = {
            "suspicious_irregular_pattern": "Higher asymmetry, darker center weighting, and irregular coverage increased this score.",
            "uniform_benign_like_pattern": "Lower asymmetry and more even visual structure increased this score.",
            "indeterminate_pigmented_pattern": "The image shows some pigment signal but not a stable enough pattern for stronger confidence.",
            "low_quality_capture": "Low contrast or softness reduces confidence in any lesion-specific interpretation.",
        }

        top_predictions = [
            VisionPrediction(
                label=label,
                confidence=confidence,
                rationale=rationales[label],
            )
            for label, confidence in ordered
        ]
        predicted_label, predicted_confidence = ordered[0]
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
        label_summaries = {
            "suspicious_irregular_pattern": "The image shows a darker, more irregular pattern that warrants cautious interpretation and in-person review.",
            "uniform_benign_like_pattern": "The image appears more uniform visually, but this should not be treated as a diagnosis.",
            "indeterminate_pigmented_pattern": "The image contains pigment structure, but the pattern is not strong enough for a confident visual interpretation.",
            "low_quality_capture": "Image quality is the main limiting factor, so the result should be treated as unreliable.",
        }
        caution = (
            "This Phase 4 vision result is a demo heuristic analysis, not a trained diagnostic classifier. "
            "Use it as an interface and explainability preview only."
        )
        if not quality.usable:
            caution += " The capture quality issues are significant enough that a clinician review is preferred."
        return label_summaries[label], caution

    def analyze(self, payload: bytes, content_type: str | None) -> ImageAnalysis:
        self.validate_upload(content_type, payload)
        image = self.load_image(payload)
        self.save_upload(image)

        metrics = self.analyze_metrics(image)
        label, top_predictions, confidence = self.classify(metrics)
        quality = self.quality(metrics)
        summary, caution = self.summarize(label, quality)
        overlay = self.overlay(image)

        return ImageAnalysis(
            analysisType="demo_heuristic",
            modelName=MODEL_NAME,
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
