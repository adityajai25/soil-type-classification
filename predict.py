from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.applications.inception_v3 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt

# Define the batch size
batch_size = 32  # You can adjust this value based on your needs

# Define or load your test generator
test_datagen = ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(
    r'/Users/adityajp/Desktop/soil type/train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load the trained model
loaded_model = load_model('alexnet_model.h5')  # Replace with the actual path to your model file

# Load and preprocess a new image for prediction
img_path = 'Black.jpg' # Replace with the path to your new image

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions
predictions = loaded_model.predict(img_array)
print("Predictions:", predictions)


# Interpret the results
predicted_class_index = np.argmax(predictions)
print("Predicted class index:", predicted_class_index)
class_labels = test_generator.class_indices
print("Class labels:", class_labels)

predicted_class_label = [k for k, v in class_labels.items() if v == predicted_class_index][0]

# Display the image with the predicted class label
plt.imshow(img)
plt.title(f'Predicted class: {predicted_class_label}')
plt.show()

# Print the predictions
print("Predicted class label:", predicted_class_label)
print("Predicted probabilities:", predictions)