import tensorflow as tf
import numpy as np
import cv2

# Load the trained generator model
from model import generator

# Load the generator model
gen_model = generator()

# Load the trained weights
gen_model.load_weights("generator_weights.h5")  # Replace "generator_weights.h5" with the path to your trained weights


# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return image


# Function to postprocess image
def postprocess_image(image):
    image = (image * 255.0).clip(0, 255).astype(np.uint8)
    return image


# Function to enhance image
def enhance_image(image):
    # Preprocess the input image
    preprocessed_image = preprocess_image(image)

    # Add batch dimension
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    # Enhance image using the generator model
    enhanced_image = gen_model.predict(preprocessed_image)

    # Remove batch dimension
    enhanced_image = np.squeeze(enhanced_image, axis=0)

    # Postprocess the enhanced image
    postprocessed_image = postprocess_image(enhanced_image)

    return postprocessed_image


# Example usage
input_image_path = "input_image.jpg"  # Replace "input_image.jpg" with the path to your input image
output_image_path = "output_image.jpg"  # Specify the path to save the output image

# Enhance the input image
enhanced_image = enhance_image(input_image_path)

# Save the enhanced image
cv2.imwrite(output_image_path, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
print("Enhanced image saved successfully!")
