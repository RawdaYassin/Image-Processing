

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

