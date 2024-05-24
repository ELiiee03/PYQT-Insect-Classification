from keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from keras.preprocessing import image # type: ignore
import numpy as np

# Load the pre-trained ResNet-50 model
model = ResNet50(weights='imagenet')

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make the prediction
    preds = model.predict(x)
    return preds
