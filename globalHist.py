import numpy as np

# Histogram helper functions

def calculate_histogram(image, width, height):
    histogram = [0] * 256

    for y in range(height):
        for x in range(width):
            pixel_value = image[y][x]
            histogram[pixel_value] += 1

    return histogram


def calculate_cumulative_histogram(histogram):
    cumulative_histogram = [0] * 256
    cumulative_sum = 0

    for i in range(256):
        cumulative_sum += histogram[i]
        cumulative_histogram[i] = cumulative_sum

    return cumulative_histogram


# global histogram equalization

def globalHistEqual(f):
    """
    Perform Global Histogram Equalization on an input image.
    @Param:
    1. f - input image as a 2D nd.array
    @Return
    - Transformed image after global histogram equalization as nd.array.
    """
    histogram = calculate_histogram(f, f.shape[1], f.shape[0])
    cumulative_histogram = calculate_cumulative_histogram(histogram)

    total_pixels = f.size
    normalized_histogram = [int(round((cumulative_histogram[i] / total_pixels) * 255)) for i in range(256)]

    equalized_image = np.array([[normalized_histogram[pixel] for pixel in row] for row in f], dtype=np.uint8)

    return equalized_image