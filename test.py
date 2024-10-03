import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model (assuming you saved it as skin_conditions_model.keras in train.py)
model = tf.keras.models.load_model('skin_conditions_model.keras')

# Path to the image (since it's in the same folder as test.py)
image_path = '/home/hot-coffee/skinqube/skinqube/Untitled.jpg'

# Load and preprocess the image
image_size = (150, 150)  # Same size as used during training
img = load_img(image_path, target_size=image_size)  # Load and resize image
img_array = img_to_array(img)  # Convert to array
img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
img_array /= 255.0  # Normalize to match training preprocessing

# Predict the class
predictions = model.predict(img_array)[0]  # Get the prediction array for the image

# Class labels based on your dataset
class_labels = ['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Milia', 'Rosacea']

# Get the indices of the top two predictions
top_indices = np.argsort(predictions)[-2:][::-1]  # Get top 2 indices in descending order
top_percentages = predictions[top_indices] * 100  # Get the corresponding percentages

# Check if the top two classes have close percentages
threshold = 5.0  # Define the threshold
if abs(top_percentages[0] - top_percentages[1]) <= threshold:
    print(f"Highest predicted class: {class_labels[top_indices[0]]} ({top_percentages[0]:.2f}%)")
    print(f"But you can also have: {class_labels[top_indices[1]]} ({top_percentages[1]:.2f}%)")
else:
    print(f"Highest predicted class: {class_labels[top_indices[0]]} ({top_percentages[0]:.2f}%)")

# Output the percentages for all classes
print("\nClass predictions:")
for i, label in enumerate(class_labels):
    percentage = predictions[i] * 100
    print(f"{label}: {percentage:.2f}%")
