

import numpy as np
from PIL import Image
import histogram

def threshold_image_array(in_image, out_image, hi, low, value, rows, cols):
    for i in range(rows):
        for j in range(cols):
            if low <= in_image[i][j] <= hi:
                out_image[i][j] = value
            else:
                out_image[i][j] = 0


# Find local maxima (peaks) in the histogram with minimum distance `peak_space`.
def find_peaks(histogram, peak_space):
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


def peaks_high_low(histogram, peak1, peak2, gray_levels):
    if peak1 is None or peak2 is None:
        return None, None

    mid_point = (peak1 + peak2) // 2

    sum1 = sum(histogram[:mid_point])
    sum2 = sum(histogram[mid_point:])

    # If the left half sum is greater, consider it as low range
    if sum1 >= sum2:
        low, hi = mid_point, gray_levels - 1
    else:
        low, hi = 0, mid_point

    # Enforce range constraints for 8-bit grayscale images
    low = max(0, low)
    hi = min(gray_levels - 1, hi)

    return hi, low



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


def threshold_and_find_means(in_image, out_image, hi, low, value, object_mean, background_mean, rows, cols):
    """
    Thresholds an input image array and produces a binary output image array.
    If the pixel in the input array is between the hi and low values, then it is set to value.
    Otherwise, it is set to 0.
    It also calculates the mean pixel intensity for the object and background.
    """
    counter = 0
    object_sum = np.int64(0)
    background_sum = np.int64(0)

    if rows == 0 or cols == 0:
        return  # Handle empty image case

    # Iterate over the image dimensions
    for i in range(rows):
        for j in range(cols):
            pixel_value = in_image[i, j]

            if low <= pixel_value <= hi:
                out_image[i, j] = value
                counter += 1
                object_sum += pixel_value
            else:
                out_image[i, j] = 0
                background_sum += pixel_value

    # Calculate object and background means
    object_mean_value = object_sum / counter if counter > 0 else 0
    background_count = (rows * cols) - counter
    background_mean_value = background_sum / background_count if background_count > 0 else 0

    # Update the object_mean and background_mean references
    object_mean[0] = int(object_mean_value)
    background_mean[0] = int(background_mean_value)

    # Debug print statements
    # print(f"\n\tTAFM> set {counter} points")
    # print(f"\n\tTAFM> object={object_mean[0]} background={background_mean[0]}")

    # Optionally return means if needed
    # return object_mean_value, background_mean_value

# Perform adaptive threshold segmentation on an image.
def adaptive_threshold_segmentation(the_image, out_image, value, segment, rows, cols, peak_space, gray_levels=256):

    # Perform histogram equalization and smoothing
    equalized_image= histogram.histogram_equalization(the_image)
    equalized_histogram, _, _ = histogram.create_histogram(
        np.array(equalized_image).flatten())
    histogram.smooth_histogram(equalized_histogram, gray_levels)

    # Use the smoothed equalized histogram for peak detection
    peak1, peak2 = find_peaks(equalized_histogram, peak_space)
    if peak1 is None or peak2 is None:
        # print("Error: Insufficient peaks found.")
        return

    hi, low = peaks_high_low(equalized_histogram, peak1, peak2, gray_levels)

    object_mean = [0]  # To store object mean (initially set to 0)
    background_mean = [0]  # To store background mean (initially set to 0)

    threshold_and_find_means(
        the_image, out_image, hi, low, value, object_mean, background_mean, rows, cols
    )

    # print(f"Object mean: {object_mean}, Background mean: {
    #       background_mean}")  # Debug: Print means

    hi, low = peaks_high_low(
        equalized_histogram, object_mean[0], background_mean[0], gray_levels)
    # Debug: Print updated thresholds
    #print(f"Updated Thresholds: hi={hi}, low={low}")
    if segment == 1:
        grow(out_image, value, rows, cols)
        
    threshold_image_array(the_image, out_image, hi, low, value, rows, cols)


# def peak_threshold_segmentation(the_image, out_image, value, rows, cols, segment, peak_space, gray_levels=256):

#     # Perform histogram equalization and smoothing
#     equalized_image = histogram.histogram_equalization(the_image)
#     equalized_histogram, _, _ = histogram.create_histogram(
#         np.array(equalized_image).flatten())
#     histogram.smooth_histogram(equalized_histogram, gray_levels)

#     # Use the smoothed equalized histogram for peak detection
#     peak1, peak2 = find_peaks(equalized_histogram, peak_space)
#     if peak1 is None or peak2 is None:
#         print("Error: Insufficient peaks found.")
#         return

#     hi, low = peaks_high_low(equalized_histogram, peak1, peak2, gray_levels)
#     if segment == 1:
#         grow(out_image, value, rows, cols)
        
#     threshold_image_array(the_image, out_image, hi, low, value, rows, cols)


def peak_threshold_segmentation(the_image, out_image, value,segment,  rows, cols, peak_space, gray_levels=256):

    # Perform histogram equalization and smoothing
    equalized_image = histogram.histogram_equalization(the_image)
    equalized_histogram, _, _ = histogram.create_histogram(
        np.array(equalized_image).flatten())
    histogram.smooth_histogram(equalized_histogram, gray_levels)

    # Use the smoothed equalized histogram for peak detection
    peak1, peak2 = find_peaks(equalized_histogram, peak_space)
    if peak1 is None or peak2 is None:
        print("Error: Insufficient peaks found.")
        return

    hi, low = peaks_high_low(equalized_histogram, peak1, peak2, gray_levels)
    #out_image = np.where(low <= the_image <= hi, value, 0).astype(np.uint8)
    threshold_image_array(the_image, out_image, hi, low, value, rows, cols)



# Segments an image using thresholding given the high and low threshold values.
def manual_threshold_segmentation(the_image, hi, low, value, segment):
    # Threshold the image array
    out_image = np.zeros_like(the_image, dtype=np.int32)
    rows, cols = the_image.shape
    #threshold_image_array(the_image, out_image, hi, low, value, rows, cols)
    
    # Perform segmentation if segment parameter is set to 1
    if segment == 1:
        grow(out_image, value, rows, cols)

    threshold_image_array(the_image, out_image, hi, low, value, rows, cols)

    return out_image.astype(np.uint8)

def valley_threshold_segmentation(the_image, out_image, value, segment, rows, cols, peak_space, gray_levels=256):
    # Perform histogram equalization and smoothing
    equalized_image= histogram.histogram_equalization(the_image)
    equalized_histogram, _, _ = histogram.create_histogram(
        np.array(equalized_image).flatten())
    histogram.smooth_histogram(equalized_histogram, gray_levels)

    # Use the smoothed equalized histogram for peak detection
    peak1, peak2 = find_peaks(equalized_histogram, peak_space)
    hi, low = valley_high_low(equalized_histogram, peak1, peak2, gray_levels)

    #threshold_image_array(the_image, out_image, hi, low, value, rows, cols)
    if segment == 1:
        grow(out_image, value, rows, cols)
        
    threshold_image_array(the_image, out_image, hi, low, value, rows, cols)


