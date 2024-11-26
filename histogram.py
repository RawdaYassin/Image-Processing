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
    '''
    شرح بالترتيب البيحصل
    
    This finds the minimum value and the maximum value
    This calculates the width of each bin. 
    The idea is to divide the entire range of pixel values (from min_val to max_val) into num_bins equal-width intervals.
    bin_width defines the size of each bin (or range of pixel values). 
    For example, if the pixel values range from 0 to 255, and there are 256 bins, then each bin will represent a width of 1 unit.
    '''
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
    '''
    flat is the array of values to be mapped (input values, which in our case are the pixel values in the flattened image).
    np.arange is the array of input values (the bin edges, or original pixel intensity values).
    min_val is the minimum pixel value in the image.
    (min_val + 256 * bin_width) which is effectively the range of bin edges for the pixel values.
    bin_width  is the width of each bin (determined by dividing the range of pixel values by the number of bins, typically 256).)
    cdf_normalized is the array of corresponding output values (the equalized pixel intensities, based on the CDF).
    '''
    equalized_flat = np.interp(flat, np.arange(min_val, min_val + 256 * bin_width, bin_width), cdf_normalized)#linear interpolation
    
    equalized_img_array = equalized_flat.reshape(img_array.shape)
    
    return Image.fromarray(equalized_img_array.astype(np.uint8))


image = Image.open('Screenshot 2024-07-12 002748.png')
equalized_image = histogram_equalization(image)
equalized_image.show()
