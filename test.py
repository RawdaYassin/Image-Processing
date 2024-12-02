
def threshold_image_array(in_image, out_image, hi, low, value, rows, cols):
    for i in range(rows):
        for j in range(cols):
            if low <= in_image[i][j] <= hi:
                out_image[i][j] = value
            else:
                out_image[i][j] = 0