import cv2
import numpy as np
import matplotlib.pyplot as plt

high_1 = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]], dtype=np.float32)

high_2 = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]], dtype=np.float32)

high_3 = np.array([
    [1, -2, 1],
    [-2, 5, -2],
    [1, -2, 1]], dtype=np.float32)

low_6 = np.array([
    [0, 1 / 6, 0],
    [1 / 6, 2 / 6, 1 / 6],
    [0, 1 / 6, 0]], dtype=np.float32)

low_9 = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]], dtype=np.float32)

low_16 = np.array([
    [1/16, 2/16, 1/16],
    [2 / 16, 4 / 16, 2 / 16],
    [1/16, 2 / 16, 1/16]], dtype=np.float32)

low_10 = np.array([
    [1/10,1/10, 1/10],
    [1/10, 2/10, 1/10],
    [1/10,1/10, 1/10]], dtype=np.float32)

def conv(image, mask):
    if mask == "First Mask":
        result = cv2.filter2D(image, -1, high_1)
    elif mask == "Second Mask":
        result = cv2.filter2D(image, -1, high_2)
    elif mask == "Third Mask":
        result = cv2.filter2D(image, -1, high_3)
    elif mask == "Mask 6":
        result = cv2.filter2D(image, -1, low_6)
    elif mask == "Mask 9":
        result = cv2.filter2D(image, -1, low_9)
    elif mask == "Mask 10":
        result = cv2.filter2D(image, -1, low_10)
    elif mask == "Mask 16":
        result = cv2.filter2D(image, -1, low_16)
    
    return result


def median_filter(original_image, window_size = 3):
    pad_size = window_size // 2
    padded_image = np.pad(original_image, pad_size, mode='edge')
    rows, columns = original_image.shape
    output_image = np.zeros_like(original_image, dtype=np.uint8)

    for i in range(rows):
        for j in range(columns):
            local_window = padded_image[i:i + window_size, j:j + window_size]
            output_image[i, j] = np.median(local_window)

    return output_image

