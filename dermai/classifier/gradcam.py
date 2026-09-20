"""Grad-CAM explainability for the skin-lesion CNN.

Grad-CAM highlights the regions of the input that most influenced the predicted
class by weighting a convolutional feature map with the gradient of the class
score flowing into it.

The published network is small (28x28 inputs), so its final convolution collapses
to a 1x1 spatial map. We therefore auto-select the *deepest convolution that still
has a spatial extent* (>= 2x2) as the Grad-CAM target, then upsample the resulting
heatmap to the display resolution. This keeps the explanation spatially meaningful.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _select_target_layer(model, override: str | None):
    """Pick the deepest Conv2D layer whose output has spatial extent >= 2."""
    import tensorflow as tf

    if override:
        return model.get_layer(override)

    candidate = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            shape = layer.output.shape
            # shape = (batch, h, w, channels)
            if len(shape) == 4 and (shape[1] or 0) >= 2 and (shape[2] or 0) >= 2:
                candidate = layer
    if candidate is None:
        # Fall back to the last conv layer regardless of spatial size.
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                candidate = layer
                break
    if candidate is None:
        raise ValueError("No Conv2D layer found for Grad-CAM.")
    return candidate


def compute_heatmap(
    model,
    preprocessed_batch: np.ndarray,
    class_index: int | None = None,
    target_layer: str | None = None,
) -> np.ndarray:
    """Return a normalised (0..1) 2D Grad-CAM heatmap for one image.

    ``preprocessed_batch`` must be shaped ``(1, H, W, 3)`` exactly as fed to the
    model at prediction time.
    """
    import tensorflow as tf

    layer = _select_target_layer(model, target_layer)
    target_index = model.layers.index(layer)

    # Feature extractor: input -> target conv feature map.
    feature_model = tf.keras.models.Model(model.inputs, layer.output)

    # Classifier head: feature map -> predictions, built by chaining the remaining
    # layers (same layer objects, so weights are shared). This guarantees the class
    # score is a differentiable function of the watched feature map under Keras 3.
    head_input = tf.keras.Input(shape=layer.output.shape[1:])
    x = head_input
    for downstream in model.layers[target_index + 1:]:
        x = downstream(x)
    head_model = tf.keras.models.Model(head_input, x)

    inputs = tf.convert_to_tensor(preprocessed_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_output = feature_model(inputs)
        tape.watch(conv_output)
        predictions = head_model(conv_output)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_output)
    # Global-average-pool the gradients -> per-channel importance weights.
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * weights, axis=-1)

    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap)
    if float(max_val) > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_heatmap(
    base_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Blend a Grad-CAM heatmap over the original image using a 'jet' colormap."""
    import matplotlib

    base = base_image.convert("RGB")
    width, height = base.size

    # Upsample the (small) heatmap to the display resolution.
    heat_img = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(
        (width, height), resample=Image.BICUBIC
    )
    heat = np.asarray(heat_img, dtype=np.float32) / 255.0

    colormap = matplotlib.colormaps["jet"]
    colored = colormap(heat)[:, :, :3]  # drop alpha
    colored = np.uint8(colored * 255)
    colored_img = Image.fromarray(colored, mode="RGB")

    return Image.blend(base, colored_img, alpha=alpha)
