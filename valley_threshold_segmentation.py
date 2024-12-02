import numpy as np
from PIL import Image
import histogram


# Find local maxima (peaks) in the histogram with minimum distance `peak_space`.
def find_peaks(histogram, peak_space=10):
    peaks = []
    for i in range(1, len(histogram) - 1):
        if histogram[i] > histogram[i - 1] and histogram[i] > histogram[i + 1]:
            peaks.append(i)

    # Sort peaks by their values in the histogram (highest peak first)
    sorted_peaks = sorted(peaks, key=lambda x: histogram[x], reverse=True)

    # Apply peak_space constraint
    selected_peaks = []
    for peak in sorted_peaks:
        if all(abs(peak - selected_peak) >= peak_space for selected_peak in selected_peaks):
            selected_peaks.append(peak)

        if len(selected_peaks) == 2:  # Stop after finding two peaks
            break

    # Return the two highest peaks, or None if not enough peaks are found
    if len(selected_peaks) == 2:
        return selected_peaks[0], selected_peaks[1]
    elif len(selected_peaks) == 1:
        return selected_peaks[0], None
    else:
        return None, None

# Determine high and low threshold values based on the valley between two peaks.
def valley_high_low(histogram, peak1, peak2, gray_levels):
    valley_point = find_valley_point(histogram, peak1, peak2)
    sum1 = sum(histogram[:valley_point])
    sum2 = sum(histogram[valley_point:])
    if sum1 >= sum2:
        low, hi = valley_point, gray_levels - 1
    else:
        low, hi = 0, valley_point
    return hi, low


# Find the valley point between two histogram peaks.
def find_valley_point(histogram, peak1, peak2):
    deltas = [[float('inf'), -1]
              for _ in range(5)]  # Array to store smallest delta values
    if peak1 < peak2:
        for i in range(peak1 + 1, peak2):
            delta_hist = histogram[i]
            insert_into_deltas(deltas, delta_hist, i)
    else:
        for i in range(peak2 + 1, peak1):
            delta_hist = histogram[i]
            insert_into_deltas(deltas, delta_hist, i)
    return deltas[0][1]  # The location of the smallest delta


# Insert a delta value and its location into the deltas array in ascending order.
def insert_into_deltas(deltas, value, place):
    for i in range(len(deltas)):
        if value < deltas[i][0]:
            deltas.insert(i, [value, place])
            deltas.pop()  # Remove the last element to maintain array size
            break



# Segment an image using histogram valley-based thresholding.
def valley_threshold_segmentation(the_image, out_image, value, segment, rows, cols, gray_levels=256, peak_space=10):
    # Perform histogram equalization and smoothing
    equalized_image= histogram.histogram_equalization(the_image)
    equalized_histogram, _, _ = histogram.create_histogram(
        np.array(equalized_image).flatten())
    histogram.smooth_histogram(equalized_histogram, gray_levels)

    # Use the smoothed equalized histogram for peak detection
    peak1, peak2 = find_peaks(equalized_histogram, peak_space)
    hi, low = valley_high_low(equalized_histogram, peak1, peak2, gray_levels)

    #threshold_image_array(the_image, out_image, hi, low, value, rows, cols)
    out_image = np.where(low <= the_image <= hi, value, 0).astype(np.uint8)


################################################################################################################
################################################################################################################

# # Load the image
# input_image = Image.open('../Image-Processing/Images/segmentation.png')

# # Convert the image to grayscale (if needed) and then to a NumPy array
# grayscale_image = input_image.convert('L')  # Convert to grayscale
# image_array = np.array(grayscale_image)  # Convert to NumPy array

# # Get the dimensions of the image
# rows, cols = image_array.shape

# # Create an output image with the same dimensions
# out_image = np.zeros_like(image_array)
# value = 255

# # Perform peak threshold segmentation
# valley_threshold_segmentation(image_array, out_image,
#                               value, 0, rows, cols, peak_space=5)

# # Create and show/save the thresholded image
# thresholded_image = Image.fromarray(out_image.astype(np.uint8))
# thresholded_image.show()
# thresholded_image.save('xx.jpg')
