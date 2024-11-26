import numpy as np
from PIL import Image

def halftone(image):
    img = np.array(image.convert('L'))  # Convert to grayscale
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            old_pixel = img[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            if x + 1 < w:
                img[y, x + 1] += quant_error * 7 / 16
            if y + 1 < h:
                if x > 0:
                    img[y + 1, x - 1] += quant_error * 3 / 16
                img[y + 1, x] += quant_error * 5 / 16
                if x + 1 < w:
                    img[y + 1, x + 1] += quant_error * 1 / 16
    return Image.fromarray(img)

# Load image
image = Image.open('Screenshot 2024-07-12 002748.png')
new_image = halftone(image)
new_image.show()
