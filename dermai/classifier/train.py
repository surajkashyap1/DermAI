"""Reproduce the HAM10000 7-class CNN training run.

This trains the exact architecture in :mod:`dermai.classifier.model` on the
HAM10000 ("Skin Cancer MNIST") dataset and saves the best checkpoint to
``models/best_model.h5`` — the same artifact shipped with the app.

Usage
-----
    pip install -r requirements.txt -r requirements-train.txt
    python -m dermai.classifier.train                 # auto-download via kagglehub
    python -m dermai.classifier.train --data-dir DIR  # use a local HAM10000 copy

A local copy must contain the metadata CSV (``HAM10000_metadata.csv``) and the
image folders. With kagglehub, set up Kaggle credentials first
(https://www.kaggle.com/docs/api). Training needs a GPU to finish quickly; the
published run reached ~97% classification accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dermai.classifier.model import NUM_CLASSES, build_model
from dermai.config import IMAGE_SIZE, MODEL_PATH

# HAM10000 dx codes -> the softmax index used by the app (see labels.py).
DX_TO_INDEX = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "nv": 4,
    "vasc": 5,
    "mel": 6,
}


def _resolve_data_dir(data_dir: str | None) -> Path:
    if data_dir:
        return Path(data_dir)
    import kagglehub

    print("Downloading HAM10000 via kagglehub ...")
    return Path(kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000"))


def _find_image(root: Path, image_id: str) -> Path | None:
    for candidate in root.rglob(f"{image_id}.jpg"):
        return candidate
    return None


def load_dataset(data_dir: Path, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load HAM10000 images + labels into arrays shaped for the CNN."""
    import pandas as pd
    from PIL import Image

    metadata_csv = next(data_dir.rglob("HAM10000_metadata*.csv"), None)
    if metadata_csv is None:
        raise FileNotFoundError(
            f"HAM10000_metadata.csv not found under {data_dir}."
        )
    frame = pd.read_csv(metadata_csv)

    images: list[np.ndarray] = []
    labels: list[int] = []
    for _, row in frame.iterrows():
        path = _find_image(data_dir, row["image_id"])
        if path is None:
            continue
        img = Image.open(path).convert("RGB").resize((image_size, image_size))
        images.append(np.asarray(img, dtype=np.float32))
        labels.append(DX_TO_INDEX[row["dx"]])

    x = np.stack(images)
    y = np.asarray(labels, dtype=np.int64)
    print(f"Loaded {len(x)} images across {len(set(labels))} classes.")
    return x, y


def train(data_dir: str | None, epochs: int, batch_size: int, output: Path) -> None:
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    root = _resolve_data_dir(data_dir)
    x, y = load_dataset(root, IMAGE_SIZE)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # HAM10000 is heavily imbalanced (nevi dominate) -> class weighting helps.
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(NUM_CLASSES), y=y_train
    )
    class_weight = {i: float(w) for i, w in enumerate(class_weights)}

    model = build_model(IMAGE_SIZE, NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(output), monitor="val_accuracy", save_best_only=True, save_weights_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f} (loss {val_loss:.4f})")
    print(f"Best weights saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DermAI HAM10000 CNN.")
    parser.add_argument("--data-dir", default=None, help="Local HAM10000 directory.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size, args.output)


if __name__ == "__main__":
    main()
