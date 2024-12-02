import numpy as np
from PIL import Image, ImageTk
# Constants
GRAY_LEVELS = 255
PEAKS = 5
PEAK_SPACE = 10

def threshold_image_array(in_image, hi, low, value):
    """
    Apply thresholding to an input image.
    Converts output to np.int8 type.
    """
    rows, cols = in_image.shape
    out_image = np.zeros((rows, cols), dtype=np.int16)
    mask = (in_image >= low) & (in_image <= hi)
    out_image[mask] = value
    #print(f"\n\tTIA> set {np.sum(mask)} points")
    return out_image

def grow(binary, value):
    """
    Grow regions based on connectivity.
    """
    rows, cols = binary.shape
    g_label = 2
    object_found = False

    for i in range(rows):
        for j in range(cols):
            if binary[i, j] == value:
                binary = label_and_check_neighbor(binary, g_label, i, j, value)
                object_found = True

            while np.any(binary == -1):  # Simulating stack operation
                stack_indices = np.argwhere(binary == -1)
                for pop_i, pop_j in stack_indices:
                    binary = label_and_check_neighbor(binary, g_label, pop_i, pop_j, value)
                    binary[pop_i, pop_j] = g_label

            if object_found:
                object_found = False
                g_label += 1

    #print(f"\nGROW> found {g_label} objects")
    return binary

def label_and_check_neighbor(binary_image, g_label, r, e, value):
    """
    Label connected regions and check neighbors.
    """
    rows, cols = binary_image.shape
    binary_image[r, e] = -1  # Temporary mark for stack simulation

    for i in range(max(0, r-1), min(rows, r+2)):
        for j in range(max(0, e-1), min(cols, e+2)):
            if binary_image[i, j] == value:
                binary_image[i, j] = -1  # Push onto stack simulation

    return binary_image

def manual_threshold_segmentation(the_image, hi, low, value, segment):
    """
    Perform manual threshold segmentation.
    Converts output to np.int8 type.
    """
    out_image = threshold_image_array(the_image, hi, low, value)
    if segment:
        out_image = grow(out_image, value)
    return out_image.astype(np.int8)

def calculate_histogram(image, bins=GRAY_LEVELS + 1):
    """
    Calculate histogram of the image.
    """
    return np.histogram(image, bins=bins, range=(0, GRAY_LEVELS))[0]

def smooth_histogram(histogram):
    """
    Smooth histogram using a simple moving average.
    """
    kernel = np.ones(5) / 5.0
    return np.convolve(histogram, kernel, mode='same')

def find_peaks(histogram):
    """
    Find the two most significant peaks in the histogram.
    """
    peaks = sorted([(val, idx) for idx, val in enumerate(histogram)], reverse=True)[:PEAKS]
    peaks = sorted(peaks, key=lambda x: x[1])  # Sort by index
    peak1, peak2 = peaks[0][1], peaks[1][1]
    return peak1, peak2

def adaptive_threshold_segmentation(the_image, value, segment):
    """
    Perform adaptive threshold segmentation.
    Converts output to np.int8 type.
    """
    histogram = calculate_histogram(the_image)
    histogram = smooth_histogram(histogram)
    peak1, peak2 = find_peaks(histogram)

    # Determine high and low thresholds
    mid_point = (peak1 + peak2) // 2
    hi = max(peak1, peak2)
    low = mid_point

    # Threshold and grow if needed
    out_image = threshold_image_array(the_image, hi, low, value)
    if segment:
        out_image = grow(out_image, value)
    return out_image.astype(np.int8)


# Example Usage
if __name__ == "__main__":
    # Test data: Replace with actual image input
    #test_image = np.random.randint(0, 256, (100, 100), dtype=np.int16)
    input_image = Image.open('../Image-Processing/Images/segmentation.PNG')
    # Load the image

    # Convert the image to grayscale (if needed) and then to a NumPy array
    grayscale_image = input_image.convert('L')  # Convert to grayscale
    image_array = np.array(grayscale_image)  # Convert to NumPy array

    # Create an output image with the same dimensions
    out_image = np.zeros_like(image_array, dtype=np.int32)
    # Perform peak threshold segmentation
    #out_image = manual_threshold_segmentation(image_array, 255, 225,255, 0)
    out_image = adaptive_threshold_segmentation(image_array, 255,0)

    # Create and show/save the thresholded image
    thresholded_image = Image.fromarray(out_image.astype(np.uint8))
    thresholded_image.show()
    thresholded_image.save('adaptive_thresholded_image1.jpg')
