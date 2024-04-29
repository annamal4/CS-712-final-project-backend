import numpy as np
from collections import OrderedDict


# Function definitions remain the same, but remove the code related to image processing

def localHistEqual4e(f, m, n):
    """
    Perform Local Histogram Equalization on a neighboring set of pixels.
    @Param:
    1. f - input image as an nd.array
    2. m - neighborhood width
    3. n - neighborhood height
    @Return
    - Transformation values on the original image, f as nd.array.
    """
    L = 255
    padded = replicate_padding(f)
    output_shape = tuple(np.array(padded.shape) - np.array([m, n]) + 1)  # reshape later to this format
    sliced = get_slices(padded, m, n)
    outer = []
    for i, sub_array in enumerate(sliced):
        val = replace_value(sub_array, L)
        outer.append(np.uint8(val))

    return np.reshape(outer, output_shape)


def replicate_padding(arr):
    """Perform replicate padding on a numpy array."""
    new_pad_shape = tuple(np.array(arr.shape) + 2)
    padded_array = np.zeros(new_pad_shape)  # create an array of zeros with new dimensions

    # perform replication
    padded_array[1:-1, 1:-1] = arr  # result will be zero-pad
    padded_array[0, 1:-1] = arr[0]  # perform edge pad for top row
    padded_array[-1, 1:-1] = arr[-1]  # edge pad for bottom row
    padded_array.T[0, 1:-1] = arr.T[0]  # edge pad for first column
    padded_array.T[-1, 1:-1] = arr.T[-1]  # edge pad for last column

    # at this point, all values except for the 4 corners should have been replicated
    padded_array[0][0] = arr[0][0]  # top left corner
    padded_array[-1][0] = arr[-1][0]  # bottom left corner
    padded_array[0][-1] = arr[0][-1]  # top right corner
    padded_array[-1][-1] = arr[-1][-1]  # bottom right corner

    return padded_array


# Function to collect m x n slices for a padded array
def get_slices(arr, width, height):
    """Collects m (width) x n (height) slices for a padded array"""
    row, col = 0, 0
    slices = []
    for i in range(len(arr) - width + 1):  # get row
        for j in range(len(arr[i]) - height + 1):  # get column
            r = i + width
            c = j + height
            sub_array = arr[i:r, j:c]
            slices.append(sub_array)
    return np.array(slices)


def replace_value(sub_array, range_L):
    """Finds the middle value of the array and returns the CDF calculated value"""
    m, n = sub_array.shape
    midpoint = round(m / n)
    value = sub_array[midpoint, midpoint]  # find the midpoint
    pdf_dict = pdf(sub_array, m, n)
    cdf = cdf_value(pdf_dict, range_L, value)  # calculate the desired cdf
    return cdf


# Function to compute the pdf of portion of an array
def pdf(sub_array, m, n):
    """
    Computes the pdf of portion of an array, indicated by sub_array
    @Param:
    1. sub_array - m * n array
    2. m - integer value for number of rows in the sub array.
    3. n - integer value for number of columns in the sub array.
    @Return
    - values : overall distribution as probability
    """
    values = {}  # dictionary for counter
    fraction = 1 / (m * n)  # iterative fraction

    for i in range(m):  # get row
        for j in range(n):  # get column
            intensity = sub_array[i][j]
            if (intensity in values):
                values[intensity] += fraction
            else:
                values[intensity] = fraction

    _, prob = zip(*list(values.items()))
    assert (np.round(sum(prob), decimals=6) == 1.0)  # assert ∑ pdf = 1

    return dict(OrderedDict(sorted(values.items())))


# Function to calculate the Cumultive Density Function (cdf) for a particular value
def cdf_value(pdf_dict, range_L, value):
    """
    Calculate the Cumultive Density Function (cdf) for a particular value
    based on its Probability Density Function (pdf) values.
    @Param:
    1. pdf_dict - dictionary of {intensity:probability} mapping
    2. range_L - overall range, L - 1 (const)
    @Returns
    - cdf: (float) cumulitive density function value based on
    """
    cdf = 0
    count, prob = zip(*list(pdf_dict.items()))  # unwrap
    for c, x in zip(count, prob):
        cdf += x * range_L
        if (c == value):
            break
    return round(cdf)


def localHistColor(image, region_size=(50, 50)):
    """
        Perform local histogram equalization on the intensity component of an HSI image.
        """
    hsi_image = rgb_to_hsi(image)
    intensity = hsi_image[:, :, 2]

    # Calculate the number of regions in each dimension
    height, width = intensity.shape
    num_regions_x = width // region_size[1]
    num_regions_y = height // region_size[0]

    # Apply histogram equalization to each region
    for y in range(num_regions_y):
        for x in range(num_regions_x):
            # Calculate the region boundaries
            region_start_x = x * region_size[1]
            region_start_y = y * region_size[0]
            region_end_x = min(region_start_x + region_size[1], width)
            region_end_y = min(region_start_y + region_size[0], height)

            # Extract the region from the intensity component
            region = intensity[region_start_y:region_end_y, region_start_x:region_end_x]

            # Perform histogram equalization on the region
            equalized_region = localHistEqual4e(region, 3, 3)  # Call your local histogram equalization method

            # Place the equalized region back into the intensity component
            intensity[region_start_y:region_end_y, region_start_x:region_end_x] = equalized_region

    # Update the intensity component in the HSI image
    hsi_image[:, :, 2] = intensity

    # Convert the modified HSI image back to RGB
    output_image = hsi_to_rgb(hsi_image)
    return output_image


def rgb_to_hsi(rgb_image):
    """
    Convert RGB image to HSI color space.
    """
    # Normalize RGB values
    r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    r_normalized = r / 255.0
    g_normalized = g / 255.0
    b_normalized = b / 255.0

    # Calculate intensity
    intensity = (r_normalized + g_normalized + b_normalized) / 3.0

    # Calculate saturation
    minimum = np.minimum.reduce([r_normalized, g_normalized, b_normalized])
    saturation = 1 - (3.0 / (r_normalized + g_normalized + b_normalized + 1e-10)) * minimum

    # Calculate hue
    numerator = 0.5 * ((r_normalized - g_normalized) + (r_normalized - b_normalized))
    denominator = np.sqrt(
        (r_normalized - g_normalized) ** 2 + (r_normalized - b_normalized) * (g_normalized - b_normalized))
    hue = np.arccos(np.clip(numerator / (denominator + 1e-10), -1, 1))
    hue[b_normalized > g_normalized] = 2 * np.pi - hue[b_normalized > g_normalized]
    hue *= 180.0 / np.pi  # Convert radians to degrees

    # Stack HSI components
    hsi_image = np.dstack((hue, saturation, intensity))
    return hsi_image


def hsi_to_rgb(hsi_image):
    """
    Convert HSI image to RGB color space.
    """
    # Extract HSI components
    hue, saturation, intensity = hsi_image[:, :, 0], hsi_image[:, :, 1], hsi_image[:, :, 2]

    # Normalize hue to range [0, 1]
    hue_normalized = hue / 360.0

    # Convert HSI to RGB
    r, g, b = np.zeros_like(hue), np.zeros_like(hue), np.zeros_like(hue)

    # Region 1: 0 <= H < 120
    b[(0 <= hue_normalized) & (hue_normalized < 1 / 3)] = intensity[
                                                              (0 <= hue_normalized) & (hue_normalized < 1 / 3)] * (
                                                                      1 - saturation[
                                                                  (0 <= hue_normalized) & (hue_normalized < 1 / 3)])
    r[(0 <= hue_normalized) & (hue_normalized < 1 / 3)] = intensity[
                                                              (0 <= hue_normalized) & (hue_normalized < 1 / 3)] * (1 + (
                saturation[(0 <= hue_normalized) & (hue_normalized < 1 / 3)] * np.cos(
            hue[(0 <= hue_normalized) & (hue_normalized < 1 / 3)])) / (np.cos(
        np.pi / 3 - hue[(0 <= hue_normalized) & (hue_normalized < 1 / 3)]) + 1e-10))
    g[(0 <= hue_normalized) & (hue_normalized < 1 / 3)] = 3 * intensity[
        (0 <= hue_normalized) & (hue_normalized < 1 / 3)] - (r[(0 <= hue_normalized) & (hue_normalized < 1 / 3)] + b[
        (0 <= hue_normalized) & (hue_normalized < 1 / 3)])

    # Region 2: 120 <= H < 240
    r[(1 / 3 <= hue_normalized) & (hue_normalized < 2 / 3)] = intensity[(1 / 3 <= hue_normalized) & (
                hue_normalized < 2 / 3)] * (1 - saturation[(1 / 3 <= hue_normalized) & (hue_normalized < 2 / 3)])
    g[(1 / 3 <= hue_normalized) & (hue_normalized < 2 / 3)] = intensity[(1 / 3 <= hue_normalized) & (
                hue_normalized < 2 / 3)] * (1 + (
                saturation[(1 / 3 <= hue_normalized) & (hue_normalized < 2 / 3)] * np.cos(
            hue[(1 / 3 <= hue_normalized) & (hue_normalized < 2 / 3)] - 2 * np.pi / 3)) / (np.cos(
        np.pi / 3 - (hue[(1 / 3 <= hue_normalized) & (hue_normalized < 2 / 3)] - 2 * np.pi / 3)) + 1e-10))
    b[(1 / 3 <= hue_normalized) & (hue_normalized < 2 / 3)] = 3 * intensity[
        (1 / 3 <= hue_normalized) & (hue_normalized < 2 / 3)] - (r[(1 / 3 <= hue_normalized) & (
                hue_normalized < 2 / 3)] + g[(1 / 3 <= hue_normalized) & (hue_normalized < 2 / 3)])

    # Region 3: 240 <= H < 360
    g[(2 / 3 <= hue_normalized) & (hue_normalized <= 1)] = intensity[
                                                               (2 / 3 <= hue_normalized) & (hue_normalized <= 1)] * (
                                                                       1 - saturation[
                                                                   (2 / 3 <= hue_normalized) & (hue_normalized <= 1)])
    b[(2 / 3 <= hue_normalized) & (hue_normalized <= 1)] = intensity[
                                                               (2 / 3 <= hue_normalized) & (hue_normalized <= 1)] * (
                                                                       1 + (saturation[(2 / 3 <= hue_normalized) & (
                                                                           hue_normalized <= 1)] * np.cos(hue[(
                                                                                                                          2 / 3 <= hue_normalized) & (
                                                                                                                          hue_normalized <= 1)] - 4 * np.pi / 3)) / (
                                                                                   np.cos(np.pi / 3 - (hue[(
                                                                                                                       2 / 3 <= hue_normalized) & (
                                                                                                                       hue_normalized <= 1)] - 4 * np.pi / 3)) + 1e-10))
    r[(2 / 3 <= hue_normalized) & (hue_normalized <= 1)] = 3 * intensity[
        (2 / 3 <= hue_normalized) & (hue_normalized <= 1)] - (g[(2 / 3 <= hue_normalized) & (hue_normalized <= 1)] + b[
        (2 / 3 <= hue_normalized) & (hue_normalized <= 1)])

    # Clip RGB values to range [0, 1]
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)

    # Stack RGB components
    rgb_image = np.dstack((r * 255, g * 255, b * 255))
    return rgb_image.astype(np.uint8)
