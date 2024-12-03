import numpy as np

def match_image_dimensions(image1, image2):
    """Adjusts two images to match their dimensions by padding the smaller one."""
    rows1, cols1 = image1.shape
    rows2, cols2 = image2.shape

    max_rows = max(rows1, rows2)
    max_cols = max(cols1, cols2)

    # Pad images with zeros to match the larger dimensions
    padded_image1 = np.zeros((max_rows, max_cols), dtype=image1.dtype)
    padded_image2 = np.zeros((max_rows, max_cols), dtype=image2.dtype)

    padded_image1[:rows1, :cols1] = image1
    padded_image2[:rows2, :cols2] = image2

    return padded_image1, padded_image2


def add_images(image1, image2):
    """Adds two grayscale images together, handling overflow."""
    image1, image2 = match_image_dimensions(image1, image2)

    # Perform addition and clip values to stay within [0, 255]
    output_image = np.clip(image1.astype(np.int16) + image2.astype(np.int16), 0, 255).astype(np.uint8)

    return output_image


def subtract_images(image1, image2):
    """Subtracts one grayscale image from another, handling underflow."""
    image1, image2 = match_image_dimensions(image1, image2)

    # Perform subtraction and clip values to stay within [0, 255]
    output_image = np.clip(image1.astype(np.int16) - image2.astype(np.int16), 0, 255).astype(np.uint8)

    return output_image


def invert_image(image):
    """Inverts a grayscale image."""
    # Inversion operation (255 - pixel value)
    output_image = (255 - image).astype(np.uint8)

    return output_image
