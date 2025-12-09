import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from matplotlib import cm
import os

def generate_gradcam(img_path,  output_dir="gradcam_outputs", img_size=(224, 224), alpha=0.4):
    """
    Generates Grad-CAM visualization for a given image and model.
    Returns the path to the saved Grad-CAM image.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = tf.keras.applications.MobileNetV2(weights="imagenet")

    # Automatically detect the last Conv2D layer
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    if last_conv is None:
        raise ValueError("No Conv2D layer found in the model!")

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=img_size)
    x = np.expand_dims(image.img_to_array(img), axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Grad-CAM model setup
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv).output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        class_idx = tf.argmax(preds[0])
        grads = tape.gradient(preds[:, class_idx], conv_out)

    # Generate heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled_grads[..., tf.newaxis])
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    # Overlay on original image
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")(np.arange(256))[:, :3]
    jet_heatmap = image.array_to_img(jet[heatmap])
    jet_heatmap = jet_heatmap.resize(img.size)
    jet_heatmap = image.img_to_array(jet_heatmap)
    superimposed = image.array_to_img(jet_heatmap * alpha + image.img_to_array(img))

    # Save result
    gradcam_path = os.path.join(output_dir, os.path.basename(img_path).replace(".png", "_gradcam.png"))
    superimposed.save(gradcam_path)

    return gradcam_path
