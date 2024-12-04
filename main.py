from PIL import Image, ImageDraw

def load_image(image_path):
    """Loads a grayscale image and returns pixel data as a list of lists."""
    try:
        img = Image.open(image_path).convert('L') #Ensure grayscale
        width, height = img.size
        pixels = []
        for y in range(height):
            row = []
            for x in range(width):
                gray = img.getpixel((x, y))
                row.append(gray)
            pixels.append(row)
        return pixels, width, height
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def save_image(pixels, width, height, path):
    """Saves pixel data as a grayscale image."""
    img = Image.new('L', (width, height))
    for y in range(height):
        for x in range(width):
            img.putpixel((x, y), pixels[y][x])
    img.save(path)

def make_copy(pixels):
    """Creates a deep copy of pixel data."""
    return [row[:] for row in pixels]

def add_images(pixels1, pixels2, width, height):
    """Adds two grayscale images together."""
    result = []
    for y in range(height):
        row = []
        for x in range(width):
            gray = min(pixels1[y][x] + pixels2[y][x], 255)
            row.append(gray)
        result.append(row)
    return result

def subtract_images(pixels1, pixels2, width, height):
    """Subtracts two grayscale images."""
    result = []
    for y in range(height):
        row = []
        for x in range(width):
            gray = max(pixels1[y][x] - pixels2[y][x], 0)
            row.append(gray)
        result.append(row)
    return result

def invert_image(pixels, width, height):
    """Inverts a grayscale image."""
    result = []
    for y in range(height):
        row = []
        for x in range(width):
            gray = 255 - pixels[y][x]
            row.append(gray)
        result.append(row)
    return result

# Load the grayscale image (replace 'image.jpg' with your actual filename)
image_path = 'image.jpg'
pixels, width, height = load_image(image_path)

if pixels:
    # Make a copy of the image
    image_copy_pixels = make_copy(pixels)
    save_image(image_copy_pixels, width, height, 'image_copy.png')

    # Add the original image and its copy
    added_image_pixels = add_images(pixels, image_copy_pixels, width, height)
    save_image(added_image_pixels, width, height, 'added_image.png')

    # Subtract the copy from the original image
    subtracted_image_pixels = subtract_images(pixels, image_copy_pixels, width, height)
    save_image(subtracted_image_pixels, width, height, 'subtracted_image.png')

    # Invert the original image
    inverted_image_pixels = invert_image(pixels, width, height)
    save_image(inverted_image_pixels, width, height, 'inverted_image.png')