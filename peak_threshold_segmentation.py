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


def peak_threshold_segmentation(the_image, out_image, value, rows, cols, gray_levels=256, peak_space=10):

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
    out_image = np.where(low <= the_image <= hi, value, 0).astype(np.uint8)
    #threshold_image_array(the_image, out_image, hi, low, value, rows, cols)


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
# peak_threshold_segmentation(image_array, out_image,
#                             value, rows, cols, peak_space=1)

# # Create and show/save the thresholded image
# thresholded_image = Image.fromarray(out_image.astype(np.uint8))
# thresholded_image.show()
# thresholded_image.save('ggggg.jpg')
