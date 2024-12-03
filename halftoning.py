
import numpy as np

def calculate_threshold(original_image):
    threshold = int(np.mean(original_image))
    return threshold


def simple_threshold(original_image):
    threshold = calculate_threshold(original_image)
    output_image = np.where(original_image > threshold, 255, 0).astype(np.uint8)
    return output_image


def error_diffusion(original_image):
    threshold = calculate_threshold(original_image)
    rows, columns = original_image.shape
    original_image = original_image.astype(np.float32)
    output_image = np.zeros_like(original_image, dtype=np.uint8)

    for y in range(rows):
        for x in range(columns):
            old_pixel = original_image[y, x]
            new_pixel = 255 if old_pixel > threshold else 0
            output_image[y, x] = new_pixel
            quantity_error = old_pixel - new_pixel
            if x + 1 < columns:
                original_image[y, x + 1] += quantity_error * 7 / 16
            if y + 1 < rows:
                if x > 0:
                    original_image[y + 1, x - 1] += quantity_error * 3 / 16
                original_image[y + 1, x] += quantity_error * 5 / 16
                if x + 1 < columns:
                    original_image[y + 1, x + 1] += quantity_error * 1 / 16
    return output_image


