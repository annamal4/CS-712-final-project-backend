import numpy as np
from collections import OrderedDict


def local_hist_equalization(f, m, n):
    print(f"Original Min/Max: {np.min(f)}, {np.max(f)}")
    L = 255
    padded = replicate_padding(f)
    output_shape = tuple(np.array(padded.shape) - np.array([m, n]) + 1)
    sliced = get_slices(padded, m, n)
    outer = []
    for sub_array in sliced:
        val = replace_value(sub_array, L)
        outer.append(val)

    equalized_image = np.reshape(outer, output_shape)
    # Normalize if necessary
    equalized_image = np.clip(equalized_image, 0, 255)  # Ensure values are within the uint8 range
    print(f"Equalized Min/Max: {np.min(equalized_image)}, {np.max(equalized_image)}")
    return equalized_image.astype(np.uint8)  # Explicit type conversion to uint8


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
    """Calculate the new value using the CDF from the PDF of sub_array."""
    m, n = sub_array.shape
    midpoint = round(m / n)
    value = sub_array[midpoint, midpoint]
    pdf_dict = pdf(sub_array, m, n)
    cdf = cdf_value(pdf_dict, range_L, value)
    return np.clip(cdf, 0, 255)  # Clip the result to ensure it doesn't go out of bounds


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
            if intensity in values:
                values[intensity] += fraction
            else:
                values[intensity] = fraction

    _, prob = zip(*list(values.items()))
    assert (np.round(sum(prob), decimals=6) == 1.0)  # assert ∑ pdf = 1

    return dict(OrderedDict(sorted(values.items())))


def cdf_value(pdf_dict, range_L, value):
    cdf = 0
    cumulative_prob = 0  # Track cumulative probability
    sorted_keys = sorted(pdf_dict.keys())
    for intensity in sorted_keys:
        cumulative_prob += pdf_dict[intensity]
        cdf = cumulative_prob * range_L
        # print(f"Intensity: {intensity}, PDF: {pdf_dict[intensity]}, Cumulative Prob: {cumulative_prob}, CDF: {cdf}")
        if intensity == value:
            break
    cdf_final = min(round(cdf), 255)
    # print(f"Final CDF for value {value}: {cdf_final}")
    return cdf_final


def local_hist_color(image, region_size=(50, 50)):
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
            equalized_region = local_hist_equalization(region, 3, 3)

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
    Convert HSI image to RGB color space using vectorized operations.
    """
    h, s, i = hsi_image[..., 0], hsi_image[..., 1], hsi_image[..., 2]
    h = h * 2 * np.pi  # Convert hue to radians

    # Initialize RGB channels
    r, g, b = np.zeros_like(i), np.zeros_like(i), np.zeros_like(i)

    # Calculate chroma
    c = (1 - np.abs(2 * i - 1)) * s
    x = c * (1 - np.abs((h / (np.pi / 3)) % 2 - 1))
    m = i - c / 2

    # Conditions for different hue segments
    mask = (0 <= h) & (h < np.pi / 3)
    r[mask], g[mask], b[mask] = c[mask], x[mask], 0
    mask = (np.pi / 3 <= h) & (h < 2 * np.pi / 3)
    r[mask], g[mask], b[mask] = x[mask], c[mask], 0
    mask = (2 * np.pi / 3 <= h) & (h < np.pi)
    r[mask], g[mask], b[mask] = 0, c[mask], x[mask]
    mask = (np.pi <= h) & (h < 4 * np.pi / 3)
    r[mask], g[mask], b[mask] = 0, x[mask], c[mask]
    mask = (4 * np.pi / 3 <= h) & (h < 5 * np.pi / 3)
    r[mask], g[mask], b[mask] = x[mask], 0, c[mask]
    mask = (5 * np.pi / 3 <= h) & (h <= 2 * np.pi)
    r[mask], g[mask], b[mask] = c[mask], 0, x[mask]

    # Combine adjusted RGB components and normalize to [0, 255]
    rgb_image = np.clip((r + m) * 255, 0, 255).astype(np.uint8)
    g = np.clip((g + m) * 255, 0, 255).astype(np.uint8)
    b = np.clip((b + m) * 255, 0, 255).astype(np.uint8)

    # Stack RGB components
    rgb_image = np.stack((rgb_image, g, b), axis=-1)

    return rgb_image
