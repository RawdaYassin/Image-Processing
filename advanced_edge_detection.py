import numpy as np
from scipy.ndimage import convolve
import basic_edge_detection
import masks
import cv2

# Homogeneity Operator
# Difference Operator
# Difference of Gaussians
# Contrast-Based Edge Detector
# Variance Edge Detector
# Range Edge Detector


def homogeneity(original_image, threshold):
    original_image = original_image.astype(np.int16)  # Avoid overflow
    rows, columns = original_image.shape
    output_image = np.zeros_like(original_image, dtype=np.uint8)

    for i in range(1, rows - 1):  # Avoid edges
        for j in range(1, columns - 1):
            local_window = original_image[i-1:i+2, j-1:j+2]
            max_diff = np.max(np.abs(local_window - original_image[i, j]))
            output_image[i, j] = 255 if max_diff > threshold else 0

    return output_image

def difference_edge(original_image, threshold):
    original_image = original_image.astype(np.int16)
    rows, columns = original_image.shape
    output_image = np.zeros_like(original_image, dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, columns - 1):
            diffs = [
                abs(original_image[i-1, j-1] - original_image[i+1, j+1]),
                abs(original_image[i-1, j+1] - original_image[i+1, j-1]),
                abs(original_image[i, j-1] - original_image[i, j+1]),
                abs(original_image[i-1, j] - original_image[i+1, j])
            ]
            max_diff = max(diffs)
            output_image[i, j] = 255 if max_diff > threshold else 0

    return output_image

def gaussian_difference(original_image, threshold, gaussian_mask):
    original_image = original_image.astype(np.int16)
    output_image = np.zeros_like(original_image, dtype=np.int8)
    if gaussian_mask == "7x7":
        mask = masks.gaussian_7x7
    elif gaussian_mask == "9x9":
        mask = masks.gaussian_9x9
    output_image = cv2.filter2D(original_image, -1, mask)
    output_image = np.where(output_image > threshold, 255, 0).astype(np.uint8)
    
    return output_image

def contrast_edge(original_image, detect_type, threshold):
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    smoothing_result = convolve(original_image.astype(np.float32), kernel)
    smoothing_result[smoothing_result == 0] = 1e-6  # Avoid division by zero

    # Detect edges using Sobel filter (replace if `basic_edge_detection` exists)
    edge_result = basic_edge_detection.detect_edges(original_image, detect_type, threshold)
    contrast_result = edge_result / smoothing_result
    normalized_result = (contrast_result / np.max(contrast_result)) * 255
    output_image = np.where(normalized_result > threshold, 255, 0).astype(np.uint8)

    return output_image

def range_filter(original_image, threshold, window_size=3):
    pad_size = window_size // 2
    padded_image = np.pad(original_image, pad_size, mode='edge')
    rows, columns = original_image.shape
    output_image = np.zeros_like(original_image, dtype=np.uint8)

    for i in range(rows):
        for j in range(columns):
            local_window = padded_image[i:i + window_size, j:j + window_size]
            pixel_range = np.max(local_window) - np.min(local_window)
            output_image[i, j] = 255 if pixel_range > threshold else 0

    return output_image

def variance_filter(original_image, threshold, window_size=3):
    pad_size = window_size // 2
    padded_image = np.pad(original_image, pad_size, mode='edge')
    rows, cols = original_image.shape
    output_image = np.zeros_like(original_image, dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            local_window = padded_image[i:i + window_size, j:j + window_size]
            mean = np.mean(local_window)
            variance = np.mean((local_window - mean) ** 2)
            output_image[i, j] = 255 if variance > threshold else 0

    return output_image
