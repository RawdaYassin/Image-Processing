from PIL import Image
import numpy as np

VALUE = 255

def threshold_image_array(original_image, high, low, value):
    original_image = original_image.astype(np.int16)
    rows, columns = original_image.shape
    output_image = np.zeros_like(original_image, dtype=np.uint8)
    counter = 0  # Initialize the counter for pixels set to the given value

    # Loop through each pixel in the image
    for i in range(rows):
        for j in range(columns):  # Fixed to use 'columns'
            if low <= original_image[i, j] <= high:
                output_image[i, j] = value
                counter += 1
            else:
                output_image[i, j] = 0
    return output_image


# Labels a pixel with an object label and checks its 8 neighbors.
def label_and_check_neighbor(binary_image, g_label, r, e, value, rows, cols, stack):
    # Label the current pixel
    binary_image[r, e] = g_label

    # Loop over the 8 neighbors (including diagonals)
    for i in range(r - 1, r + 2):  # Rows: r-1, r, r+1
        for j in range(e - 1, e + 2):  # Columns: e-1, e, e+1
            # Ensure (i, j) are within the bounds of the image
            if 0 <= i < rows and 0 <= j < cols:
                # If the neighboring pixel matches the 'value', add it to the stack
                if binary_image[i, j] == value:
                    stack.append((i, j))


# Detects connected objects in a binary image array and labels them.
def grow(binary_image, value, rows, cols):
    g_label = 2  # Start the labeling from 2
    stack = []  # Stack for storing coordinates of pixels to process

    for i in range(rows):
        for j in range(cols):
            # Search for the first pixel of a region
            if binary_image[i, j] == value:
                # Label the current pixel and its neighbors
                label_and_check_neighbor(binary_image, g_label, i, j, value, rows, cols, stack)

                # Process pixels in the stack
                while len(stack) > 0:
                    pop_i, pop_j = stack.pop()
                    label_and_check_neighbor(binary_image, g_label, pop_i, pop_j, value, rows, cols, stack)

                # Increment the label for the next object
                g_label += 1

    # g_label starts from 2, so subtract 2 for the actual count
    print(f"GROW> Found {g_label - 2} objects")


# Segments an image using thresholding given the high and low threshold values.
def manual_threshold_segmentation(original_image, high, low, value, segment):
    # Threshold the image array
    rows, columns = original_image.shape
    output_image = threshold_image_array(original_image, high, low, value)

    # Perform segmentation if segment parameter is set to 1
    if segment == 1:
        grow(output_image, value, rows, columns)

    return output_image


################################################################################################################
################################################################################################################

# # Load the image
# input_image = Image.open('../Image-Processing/Images/Capture.PNG')  # Adjust path as needed

# # Convert the image to grayscale (if needed) and then to a NumPy array
# grayscale_image = input_image.convert('L')  # Convert to grayscale
# image_array = np.array(grayscale_image)  # Convert to NumPy array

# # Perform peak threshold segmentation
# out_image = manual_threshold_segmentation(image_array, 255, 225, VALUE, 1)

# # Create and show the thresholded image
# thresholded_image = Image.fromarray(out_image.astype(np.uint8))
# thresholded_image.show()
