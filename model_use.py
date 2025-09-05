from PIL import Image
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import random



DATASET_PATH = 'datasets/'
num_classes = len(os.listdir(DATASET_PATH))
class_mode = 'binary' if num_classes == 2 else 'categorical'

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"error finding files at filepath {image_path}")
        return

    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except (OSError, IOError):
        print(f"Error: Corrupted image at {image_path}")
    model = tf.keras.models.load_model("imageclassifier.h5")
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Failed to read image at {image_path}")
        return

    img = cv2.resize(img, (128, 128))
    img = img / 255
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_names = os.listdir(DATASET_PATH)

    if class_mode == "binary":
        predicted_class = class_names[int(bool(prediction[0] > 0.5))]
    else:
        predicted_class = class_names[tf.argmax(prediction, axis=1).numpy()[0]]
    print(f"The model determined: {predicted_class}")

    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"The model determined: {predicted_class}")
    plt.axis('off')
    plt.show()

predict_image(f"untested/dog (12).jpg")