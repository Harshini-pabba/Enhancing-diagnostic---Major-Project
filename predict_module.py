import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load and predict on new image
def predict_new_image(img_path, model, img_size=(256,256)):
    # Load image
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Predict
    probs = model.predict(img_array)[0]
    pred_idx = np.argmax(probs)
    pred_class = class_names[pred_idx]
    confidence = np.max(probs)

    # Show image + prediction
    plt.imshow(tf.keras.utils.load_img(img_path))
    plt.axis("off")
    plt.title(f"Prediction: {pred_class} ({confidence:.2f})")
    plt.show()

    return pred_class, confidence

#from tensorflow import keras
class_names = ['Covid19', 'Normal', 'Pneumonia', 'Tuberculosis']
#model = keras.models.load_model(r"D:\MajorProjectChestXRay\MobileNetV2_chestxray_model.keras", compile=False)
#print(" Pretrained model loaded successfully.")

# Example usage (replace with your own file path)
#new_img_path = r"D:\MajorProjectChestXRay\Chest x rays_test\Tuberculosis_test\Tuberculosis-690.png"
#pred_class, conf = predict_new_image(new_img_path, model)
#print("Predicted:", pred_class, "with confidence:", conf)


