import cv2
import numpy as np
import matplotlib.pyplot as plt
import masks


# Roberts Operator
# Prewitt Operator
# Kirsch Compass Masks
# Robinson Compass Masks
# Laplacian Operators (Quick mask)



def detect_edges(original_image, detect_type, threshold):
    rows, columns = original_image.shape
    if detect_type == "LAPLACE":
        output_image = quick_edge(original_image, detect_type, threshold, rows, columns)
    else:
        output_image = perform_convolution(original_image, detect_type, threshold, rows, columns)
    fix_edges(output_image, 1, rows, columns)
    return output_image

def perform_convolution(original_image, detect_type, threshold, rows, columns):
    masks = setup_masks(detect_type)
    output_image = np.zeros_like(original_image)

    # Apply all masks at once
    results = [cv2.filter2D(original_image, -1, mask) for mask in masks]
    combined_result = np.max(results, axis=0)

    # Thresholding
    output_image[combined_result > threshold] = 255
    output_image[combined_result <= threshold] = 0

    return output_image

def quick_edge(original_image, detect_type, threshold, rows, columns):
    mask = setup_masks(detect_type)
    result = cv2.filter2D(original_image, -1, mask)

    # Thresholding
    result[result > threshold] = 255
    result[result <= threshold] = 0

    return result

def setup_masks(detect_type):
    if detect_type == 'KIRSCH':
        return masks.kirsch_mask_0, masks.kirsch_mask_1, masks.kirsch_mask_2, masks.kirsch_mask_3, masks.kirsch_mask_4, masks.kirsch_mask_5, masks.kirsch_mask_6, masks.kirsch_mask_7
    elif detect_type == "LAPLACE":
        return masks.quick_mask
    elif detect_type == 'PREWITT':
        return masks.prewitt_mask_0, masks.prewitt_mask_1, masks.prewitt_mask_2, masks.prewitt_mask_3, masks.prewitt_mask_4, masks.prewitt_mask_5, masks.prewitt_mask_6, masks.prewitt_mask_7
    elif detect_type == 'SOBEL':
        return masks.sobel_mask_0, masks.sobel_mask_1, masks.sobel_mask_2, masks.sobel_mask_3, masks.sobel_mask_4, masks.sobel_mask_5, masks.sobel_mask_6, masks.sobel_mask_7
    elif detect_type == 'ROBINSON':
        return masks.robinson_mask_0, masks.robinson_mask_1, masks.robinson_mask_2, masks.robinson_mask_3, masks.robinson_mask_4, masks.robinson_mask_5, masks.robinson_mask_6, masks.robinson_mask_7   
    else:
        raise ValueError("Unknown detection type")

def fix_edges(output_image, border_value, rows, columns):
    output_image[0, :] = output_image[-1, :] = border_value
    output_image[:, 0] = output_image[:, -1] = border_value
