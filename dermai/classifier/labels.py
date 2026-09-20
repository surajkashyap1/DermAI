"""The seven HAM10000 lesion classes with plain-language metadata.

Class order matches the label encoding used to train ``best_model.h5``
(the softmax output index -> class mapping from the original project).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassInfo:
    index: int
    code: str          # HAM10000 short code
    name: str          # human-readable name
    malignancy: str    # cancer risk framing
    description: str


CLASSES: tuple[ClassInfo, ...] = (
    ClassInfo(
        index=0,
        code="akiec",
        name="Actinic keratosis / intraepithelial carcinoma",
        malignancy="Pre-malignant / cancerous",
        description=(
            "A rough, scaly patch that develops from years of sun exposure, most "
            "often on the face, lips, ears, hands, forearms, scalp or neck. It can "
            "progress to squamous cell carcinoma and should be evaluated."
        ),
    ),
    ClassInfo(
        index=1,
        code="bcc",
        name="Basal cell carcinoma",
        malignancy="Cancerous",
        description=(
            "The most common skin cancer, typically on sun-exposed skin. It is the "
            "least dangerous form but still requires treatment."
        ),
    ),
    ClassInfo(
        index=2,
        code="bkl",
        name="Benign keratosis-like lesions",
        malignancy="Non-cancerous",
        description=(
            "Includes seborrheic keratoses, common in older adults, appearing as "
            "brown, black or light-tan spots on the face, chest, shoulders or back."
        ),
    ),
    ClassInfo(
        index=3,
        code="df",
        name="Dermatofibroma",
        malignancy="Non-cancerous",
        description=(
            "Harmless round, reddish-brown spots caused by an overgrowth of fibrous "
            "tissue."
        ),
    ),
    ClassInfo(
        index=4,
        code="nv",
        name="Melanocytic nevi",
        malignancy="Non-cancerous",
        description=(
            "Common moles containing nevus cells. Usually harmless, but changes in "
            "size, shape or colour warrant review."
        ),
    ),
    ClassInfo(
        index=5,
        code="vasc",
        name="Pyogenic granulomas / vascular lesions",
        malignancy="Usually benign (can bleed)",
        description=(
            "Small, raised, red bumps rich in blood vessels that bleed easily. "
            "Generally benign but sometimes needs removal."
        ),
    ),
    ClassInfo(
        index=6,
        code="mel",
        name="Melanoma",
        malignancy="Cancerous (most dangerous)",
        description=(
            "The most dangerous form of skin cancer. When recognised and treated "
            "early it is almost always curable, so prompt evaluation matters."
        ),
    ),
)

# index -> readable name, kept for convenience / display.
CLASS_LABELS: dict[int, str] = {c.index: c.name for c in CLASSES}


def class_info(index: int) -> ClassInfo:
    """Return the :class:`ClassInfo` for a softmax output index."""
    for info in CLASSES:
        if info.index == index:
            return info
    raise IndexError(f"No class defined for index {index}")
