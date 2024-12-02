import numpy as np
from PIL import Image
import histogram

def smooth_histogram(histogram, gray_levels):
    smoothed = np.zeros_like(histogram)
    smoothed[0] = (histogram[0] + histogram[1]) // 2
    smoothed[-1] = (histogram[-1] + histogram[-2]) // 2

    for i in range(1, gray_levels - 1):
        smoothed[i] = (histogram[i - 1] + histogram[i] + histogram[i + 1]) // 3

    for i in range(gray_levels):
        histogram[i] = smoothed[i]


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


# def threshold_and_find_means(in_image, out_image, hi, low, value, object_mean, background_mean, rows, cols):
#     """
#     Thresholds an input image array and produces a binary output image array.
#     If the pixel in the input array is between the hi and low values, then it is set to value.
#     Otherwise, it is set to 0.
#     It also calculates the mean pixel intensity for the object and background.
#     """
#     counter = 0
#     object_sum = 0
#     background_sum = 0

#     # Iterate over the image dimensions
#     for i in range(rows):
#         for j in range(cols):
#             pixel_value = in_image[i, j]

#             if low <= pixel_value <= hi:
#                 out_image[i, j] = value
#                 counter += 1
#                 object_sum += pixel_value
#             else:
#                 out_image[i, j] = 0
#                 background_sum += pixel_value

#     # Calculate object and background means
#     if counter > 0:
#         object_mean_value = object_sum / counter
#     else:
#         object_mean_value = 0

#     background_count = (rows * cols) - counter
#     if background_count > 0:
#         background_mean_value = background_sum / background_count
#     else:
#         background_mean_value = 0

#     # Update the object_mean and background_mean references
#     object_mean[0] = int(object_mean_value)
#     background_mean[0] = int(background_mean_value)

#     # Debug print statements
#     # print(f"\n\tTAFM> set {counter} points")
#     # print(f"\n\tTAFM> object={object_mean[0]} background={background_mean[0]}")

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



def threshold_image_array(in_image, out_image, hi, low, value, rows, cols):
    for i in range(rows):
        for j in range(cols):
            if low <= in_image[i][j] <= hi:
                out_image[i][j] = value
            else:
                out_image[i][j] = 0


# Perform adaptive threshold segmentation on an image.
def adaptive_threshold_segmentation(the_image, out_image, value, segment, rows, cols, gray_levels=256, peak_space=10):

    # Perform histogram equalization and smoothing
    equalized_image= histogram.histogram_equalization(the_image)
    equalized_histogram, _, _ = histogram.create_histogram(
        np.array(equalized_image).flatten())
    smooth_histogram(equalized_histogram, gray_levels)

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
    print(f"Updated Thresholds: hi={hi}, low={low}")

    threshold_image_array(the_image, out_image, hi, low, value, rows, cols)


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
# adaptive_threshold_segmentation(image_array, out_image,
#                                 value, 0, rows, cols, peak_space=10)

# # Create and show/save the thresholded image
# thresholded_image = Image.fromarray(out_image.astype(np.uint8))
# thresholded_image.show()
# thresholded_image.save('adaptive_thresholded_image.jpg')
