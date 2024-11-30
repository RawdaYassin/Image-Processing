import numpy as np
from PIL import Image


def calculate_cumulative_histogram(hist):
    cdf = np.zeros_like(hist, dtype=hist.dtype)
    cdf[0] = hist[0]

    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + hist[i]

    return cdf


def create_histogram(data, num_bins=256):
    min_val = min(data)
    max_val = max(data)
    bin_width = (max_val - min_val) / num_bins
    histogram = np.zeros(num_bins, dtype=int)

    for value in data:
        bin_index = int((value - min_val) // bin_width)

        # Ensure maximum value doesn't cause an out-of-bounds error.
        if bin_index == num_bins:
            bin_index = num_bins - 1
        histogram[bin_index] += 1

    return histogram, min_val, bin_width


def histogram_equalization(image):
    # Check if the image is a PIL Image, if not, convert it
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if image.mode != 'L':
        image = image.convert('L')  # Convert to grayscale if not already

    img_array = np.array(image)
    flat = img_array.flatten()

    # Create histogram
    hist, min_val, bin_width = create_histogram(flat, num_bins=256)

    # Calculate CDF
    cdf = calculate_cumulative_histogram(hist)
    cdf_normalized = cdf * 255 / cdf[-1]

    # Perform histogram equalization
    equalized_flat = np.interp(flat, np.arange(
        min_val, min_val + 256 * bin_width, bin_width), cdf_normalized)
    equalized_img_array = equalized_flat.reshape(img_array.shape)
    equalized_image = Image.fromarray(equalized_img_array.astype(np.uint8))

    return equalized_image, hist, cdf  # Return the image, histogram, and CDF


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


def threshold_image_array(in_image, out_image, hi, low, value, rows, cols):
    for i in range(rows):
        for j in range(cols):
            if low <= in_image[i, j] <= hi:
                out_image[i, j] = value
            else:
                out_image[i, j] = 0


def peak_threshold_segmentation(the_image, out_image, value, rows, cols, gray_levels=256, peak_space=10):

    # Perform histogram equalization and smoothing
    equalized_image, _, _ = histogram_equalization(the_image)
    equalized_histogram, _, _ = create_histogram(
        np.array(equalized_image).flatten())
    smooth_histogram(equalized_histogram, gray_levels)

    # Use the smoothed equalized histogram for peak detection
    peak1, peak2 = find_peaks(equalized_histogram, peak_space)
    if peak1 is None or peak2 is None:
        print("Error: Insufficient peaks found.")
        return

    hi, low = peaks_high_low(equalized_histogram, peak1, peak2, gray_levels)

    threshold_image_array(the_image, out_image,
                          hi, low, value, rows, cols)


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
out_image = np.zeros_like(image_array)
value = 255

# Perform peak threshold segmentation
peak_threshold_segmentation(image_array, out_image,
                            value, rows, cols, peak_space=1)

# Create and show/save the thresholded image
thresholded_image = Image.fromarray(out_image.astype(np.uint8))
thresholded_image.show()
thresholded_image.save('peak_thresholded_image.jpg')
