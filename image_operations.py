
import numpy as np

def add_images(pixels1, pixels2, width, height):
    """Adds two grayscale images together."""
    original_image = original_image.astype(np.int16)  # Avoid overflow
    rows, columns = original_image.shape
    output_image = np.zeros_like(original_image, dtype=np.uint8)
    for y in range(rows):
        for x in range(columns):
            output_image[y,x] =  min(pixels1[y][x] + pixels2[y][x], 255)
    return output_image


def subtract_images(pixels1, pixels2):
    """Subtracts two grayscale images."""
    original_image = original_image.astype(np.int16)  # Avoid overflow
    rows, columns = original_image.shape
    output_image = np.zeros_like(original_image, dtype=np.uint8)
    for y in range(rows):
        for x in range(columns):
            output_image[y,x] = max(pixels1[y][x] - pixels2[y][x], 0)
    return output_image

def invert_image(pixels):
    """Inverts a grayscale image."""
    original_image = original_image.astype(np.int16)  # Avoid overflow
    rows, columns = original_image.shape
    output_image = np.zeros_like(original_image, dtype=np.uint8)
    for y in range(rows):
        for x in range(columns):
            output_image[y,x] = 255 - pixels[y,x]
    return output_image

