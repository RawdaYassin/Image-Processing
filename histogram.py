import numpy as np
from PIL import Image

def histogram_equalization(image):

    if image.mode != 'L':
        image = image.convert('L')
    
    img_array = np.array(image)
    flat = img_array.flatten()  
    
    hist, bins = np.histogram(flat, bins=256, range=[0, 256], density=True)
    
    cdf = hist.cumsum() 
    cdf_normalized = cdf * (255 / cdf[-1])  
    
    equalized_img_array = np.interp(flat, bins[:-1], cdf_normalized).reshape(img_array.shape)
    
    return Image.fromarray(equalized_img_array.astype(np.uint8))

image = Image.open('Screenshot 2024-07-12 002748.png')

equalized_image = histogram_equalization(image)

equalized_image.save('equalized_image.png')
equalized_image.show()