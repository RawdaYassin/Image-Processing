import numpy as np
from PIL import Image

def calculate_cumulative_histogram(hist):
    cdf = np.zeros_like(hist, dtype=hist.dtype)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + hist[i]
    return cdf

# Function to create the histogram (returns counts for each bin)
def create_histogram(data, num_bins=256):
    min_val = min(data)
    max_val = max(data)
    bin_width = (max_val - min_val) / num_bins

    histogram = np.zeros(num_bins, dtype=int)  
    for value in data:
        bin_index = int((value - min_val) // bin_width)
        if bin_index == num_bins: #we want to ensure that the maximum value doesn't cause an out of bounds error. 
            bin_index = num_bins - 1
        histogram[bin_index] += 1

    return histogram, min_val, bin_width

# Histogram equalization function
def histogram_equalization(image):
    if image.mode != 'L':  
        image = image.convert('L')
    
    img_array = np.array(image) 
    flat = img_array.flatten() 
    
    # Create histogram
    hist, min_val, bin_width =create_histogram(flat, num_bins=256)

    # Calculate CDF
    cdf =calculate_cumulative_histogram(hist)
    
    cdf_normalized = cdf * 255 / cdf[-1] 

    equalized_flat = np.interp(flat, np.arange(min_val, min_val + 256 * bin_width, bin_width), cdf_normalized)#linear interpolation
    
    equalized_img_array = equalized_flat.reshape(img_array.shape)
    
    return Image.fromarray(equalized_img_array.astype(np.uint8))


