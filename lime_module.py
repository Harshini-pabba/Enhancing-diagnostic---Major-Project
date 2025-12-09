from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os



# --- Generate LIME Explanation ---
def generate_lime_explanation(img_path, output_dir="lime_outputs"):
    """
    Takes an image path, generates a LIME explanation using pretrained MobileNetV2.
    Returns path of the saved explanation image.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img).astype(np.uint8)
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image=img,
        classifier_fn=lambda x: model.predict(preprocess(x)),
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    # Generate mask
    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    # Save output image
    lime_img_path = os.path.join(output_dir, os.path.basename(img_path).replace(".png", "_lime.png"))
    plt.figure(figsize=(6, 6))
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title("LIME Explanation")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(lime_img_path, bbox_inches="tight")
    plt.close()

    return lime_img_path
