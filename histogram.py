import numpy as np
from PIL import Image
import cv2

def calculate_cumulative_histogram(hist):
    cdf = np.zeros_like(hist, dtype=hist.dtype)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + hist[i]
    return cdf

# Function to create the histogram (returns counts for each bin)
# def create_histogram(data, num_bins=256):
#     min_val = min(data)
#     max_val = max(data)
#     bin_width = (max_val - min_val) / num_bins

#     histogram = np.zeros(num_bins, dtype=int)  
#     for value in data:
#         bin_index = int((value - min_val) // bin_width)
#         if bin_index == num_bins: #we want to ensure that the maximum value doesn't cause an out of bounds error. 
#             bin_index = num_bins - 1
#         histogram[bin_index] += 1

#     return histogram, min_val, bin_width

import numpy as np

def compute_histogram(image_array):
    histogram = [256]
    histogram, bin_edges = np.histogram(image_array, bins=num_bins, range=(0, 256))
    return histogram


def histogram_to_image(original_image):
    histogram = compute_histogram(original_image)
    image_height = 256
    image_width = 256
    """
    Converts a histogram into an image representation.

    Parameters:
        histogram (np.ndarray): Array of histogram values.
        image_height (int): Height of the output image.
        image_width (int): Width of the output image.

    Returns:
        np.ndarray: Grayscale image representing the histogram.
    """
    # Normalize the histogram to fit within the image height
    hist_max = np.max(histogram)
    normalized_hist = (histogram / hist_max) * image_height

    # Create a blank white image
    hist_image = np.ones((image_height, image_width), dtype=np.uint8) * 255

    # Calculate the bin width in the image
    bin_width = image_width // len(histogram)

    for i, value in enumerate(normalized_hist):
        x_start = i * bin_width
        x_end = x_start + bin_width

        # Draw a vertical bar for the histogram bin
        cv2.rectangle(hist_image, (x_start, image_height), (x_end, image_height - int(value)), (0,), -1)

    return hist_image


# Histogram equalization function
# def histogram_equalization(image):
#     if image.mode != 'L':  
#         image = image.convert('L')
    
#     img_array = np.array(image) 
#     flat = img_array.flatten() 
    
#     # Create histogram
#     hist, min_val, bin_width =create_histogram(flat, num_bins=256)

#     # Calculate CDF
#     cdf =calculate_cumulative_histogram(hist)
    
#     cdf_normalized = cdf * 255 / cdf[-1] 

#     equalized_flat = np.interp(flat, np.arange(min_val, min_val + 256 * bin_width, bin_width), cdf_normalized)#linear interpolation
    
#     equalized_img_array = equalized_flat.reshape(img_array.shape)
    
#     return Image.fromarray(equalized_img_array.astype(np.uint8))



def histogram_equalization(original_image):

    # Flatten the image array
    flat = original_image.flatten()

    # Compute histogram and cumulative distribution function (CDF)
    histogram = np.histogram(flat, bins=256, range=(0, 256))
    cdf = np.cumsum(histogram)

    # Normalize the CDF to map values to [0, 255]
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf[-1] - cdf.min())

    # Map the original pixel values to equalized values
    equalized_flat = cdf_normalized[flat]

    # Reshape to the original image shape
    output_image = equalized_flat.reshape(original_image.shape)

    # Clip and convert to uint8
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    # Convert back to PIL image
    return output_image
