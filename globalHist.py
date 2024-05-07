import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
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

def save_histogram_image(histogram, title, save_path):
    plt.figure(figsize=(6, 4))
    plt.bar(range(256), histogram)
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

def plot_histograms(histogram, equalized_histogram):
    save_histogram_image(histogram, 'Original Histogram', "./test/original")
    save_histogram_image(equalized_histogram, 'Equalized Histogram', "./test/equalized")


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

    equalized_histogram = calculate_histogram(equalized_image, equalized_image.shape[1], equalized_image.shape[0])
    p = Process(target=plot_histograms, args=(histogram, equalized_histogram))
    p.start()
    p.join

    return equalized_image