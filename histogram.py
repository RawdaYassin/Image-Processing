import numpy as np
from PIL import Image
import matplotlib as plt
def calculate_cumulative_histogram(hist):
    cdf = np.zeros_like(hist, dtype=hist.dtype)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + hist[i]
    return cdf

# Function to create the histogram (returns counts for each bin)
def create_histogram(original_image, num_bins=256):
    original_image = original_image.flatten() 
    min_val = min(original_image)
    max_val = max(original_image)
    bin_width = (max_val - min_val) / num_bins

    histogram = np.zeros(num_bins, dtype=int)  
    for value in original_image:
        bin_index = int((value - min_val) // bin_width)
        if bin_index == num_bins: #we want to ensure that the maximum value doesn't cause an out of bounds error. 
            bin_index = num_bins - 1
        histogram[bin_index] += 1

    return histogram, min_val, bin_width

# Histogram equalization function
def histogram_equalization(original_image):
    original_image = original_image.astype(np.int16)
    output_image = np.zeros_like(original_image, dtype=np.int8)
    #img_array = np.array(image) 
    flat = original_image.flatten() 
    
    # Create histogram
    hist, min_val, bin_width =create_histogram(flat, num_bins=256)

    # Calculate CDF
    cdf =calculate_cumulative_histogram(hist)
    
    cdf_normalized = cdf * 255 / cdf[-1] 
    equalized_flat = np.interp(flat, np.arange(min_val, min_val + 256 * bin_width, bin_width), cdf_normalized)#linear interpolation
    output_image = equalized_flat.reshape(original_image.shape)
    return output_image.astype(np.uint8)

def smooth_histogram(histogram, gray_levels):
    smoothed = np.zeros_like(histogram)
    smoothed[0] = (histogram[0] + histogram[1]) // 2
    smoothed[-1] = (histogram[-1] + histogram[-2]) // 2

    for i in range(1, gray_levels - 1):
        smoothed[i] = (histogram[i - 1] + histogram[i] + histogram[i + 1]) // 3

    for i in range(gray_levels):
        histogram[i] = smoothed[i]



# image = Image.open('Screenshot 2024-07-12 002748.png')
# equalized_image = histogram_equalization(image)
# equalized_image.show()


# def display_histogram(image, num_bins=256):

#     # Compute histogram
#     histogram, min, w = create_histogram(image)
#     bins = [256]
#     # Plot histogram
#     fig, ax = plt.subplots()
#     ax.bar(bins[:-1], histogram, width=bins[1] - bins[0], color='black')
#     ax.set_xlim(0, 255)
#     ax.set_title("Histogram")
#     ax.set_xlabel("Pixel Intensity")
#     ax.set_ylabel("Frequency")
    
#     # Save the figure to a NumPy array
#     fig.canvas.draw()
#     histogram_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     histogram_array = histogram_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
#     plt.close(fig)  # Close the plot to free memory
#     return histogram_array


