import basic_edge_detection
import advanced_edge_detection
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import halftoning


# Load an image using Pillow and convert to grayscale
original_image = Image.open("Images/Capture.PNG").convert("L")
original_image = np.array(original_image)  # Convert to NumPy array

# Perform edge detection using the custom function
detect_type = 'LAPLACE'
threshold = 128  # Example threshold value

#out_image = basic_edge_detection.detect_edges(original_image, detect_type, threshold)
out_image = halftoning.halftone(original_image,threshold)

# Convert the output image (NumPy array) back to a PIL Image for display
out_image = Image.fromarray(np.uint8(out_image))

# Display results
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image, cmap='gray')

# Edge-detected image
plt.subplot(1, 2, 2)
plt.title("Edge Detected Image")
plt.imshow(out_image, cmap='gray')

plt.show()
