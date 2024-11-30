from PIL import Image
import numpy as np


def threshold_image_array(in_image, out_image, hi, low, value, rows, cols):
    counter = 0  # Initialize the counter for pixels set to the given value

    # Loop through each pixel in the image
    for i in range(rows):
        for j in range(cols):
            if low <= in_image[i][j] <= hi:
                out_image[i][j] = value
                counter += 1
            else:
                out_image[i][j] = 0


# Labels a pixel with an object label and checks its 8 neighbors.If any of the neighbors have the 'value', they are added to the stack for further labeling.
def label_and_check_neighbor(binary_image, g_label, r, e, value, first_call, rows, cols, stack):
    # Check if the current pixel has already been labeled
    if binary_image[r][e] == g_label:
        return  # Skip processing if already labeled

    # Label the current pixel
    binary_image[r][e] = g_label

    # Loop over the 8 neighbors (including diagonals)
    for i in range(r - 1, r + 2):  # Rows: r-1, r, r+1
        for j in range(e - 1, e + 2):  # Columns: e-1, e, e+1
            # Ensure (i, j) are within the bounds of the image
            if 0 <= i < rows and 0 <= j < cols:
                # If the neighboring pixel matches the 'value', add it to the stack
                if binary_image[i][j] == value:
                    stack.append((i, j))


# Detects connected objects in a binary image array and labels them.
def grow(binary_image, value, rows, cols):
    g_label = 2  # Start the labeling from 2
    object_found = False
    first_call = True
    stack = []  # Stack for storing coordinates of pixels to process

    for i in range(rows):
        for j in range(cols):
            # Search for the first pixel of a region
            if binary_image[i][j] == value:
                label_and_check_neighbor(
                    binary_image, g_label, i, j, value, first_call, rows, cols, stack)
                object_found = True

            # Process pixels in the stack if it's not empty
            while len(stack) > 0:
                pop_i, pop_j = stack.pop()
                label_and_check_neighbor(
                    binary_image, g_label, pop_i, pop_j, value, first_call, rows, cols, stack)

            # If an object was found, increment the label
            if object_found:
                object_found = False
                g_label += 1

    # g_label starts from 2, so subtract 2 for the actual count
    print(f"GROW> found {g_label - 2} objects")


# Segments an image using thresholding given the high and low threshold values.
def manual_threshold_segmentation(the_image, out_image, hi, low, value, segment, rows, cols):
    # Threshold the image array
    threshold_image_array(the_image, out_image, hi, low, value, rows, cols)

    # Perform segmentation if segment parameter is set to 1
    if segment == 1:
        grow(out_image, value, rows, cols)


################################################################################################################
################################################################################################################

# Load the image
input_image = Image.open('Capture.png')

# Convert the image to grayscale (if needed) and then to a NumPy array
grayscale_image = input_image.convert('L')  # Convert to grayscale
image_array = np.array(grayscale_image)  # Convert to NumPy array

# Get the dimensions of the image
rows, cols = image_array.shape

# Create an output image with the same dimensions
out_image = np.zeros_like(image_array, dtype=np.int32)
value = 255

# Perform peak threshold segmentation
manual_threshold_segmentation(image_array, out_image, 255, 225,
                              value, 0, rows, cols)

# Create and show/save the thresholded image
thresholded_image = Image.fromarray(out_image.astype(np.uint8))
thresholded_image.show()
thresholded_image.save('manual_thresholded_image.jpg')
