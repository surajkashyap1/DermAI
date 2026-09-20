"""The HAM10000 7-class CNN architecture.

This reproduces the published DermAI classifier: a compact convolutional network
on 28x28 RGB inputs with BatchNormalization after pooling/dense blocks, which was
found to stabilise validation accuracy. ``best_model.h5`` holds the trained
weights for exactly this graph.
"""

from __future__ import annotations

from dermai.config import IMAGE_SIZE

NUM_CLASSES = 7


def build_model(image_size: int = IMAGE_SIZE, num_classes: int = NUM_CLASSES):
    """Construct (uncompiled) the 7-class skin-lesion CNN.

    Kept import-light: TensorFlow is imported lazily so that modules which only
    need label metadata (or the RAG stack) do not pay the TF import cost.
    """
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv2D,
        Dense,
        Dropout,
        Flatten,
        Input,
        MaxPool2D,
    )
    from tensorflow.keras.models import Sequential

    model = Sequential(name="dermai_ham10000_cnn")
    # Explicit Input layer: builds the model at construction time so `model.input`
    # / `model.output` are defined (needed for Grad-CAM under Keras 3). It carries
    # no weights, so the weighted-layer order — and thus best_model.h5 — is unchanged.
    model.add(Input(shape=(image_size, image_size, 3)))
    model.add(
        Conv2D(
            16,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )
    )
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation="softmax"))
    return model
