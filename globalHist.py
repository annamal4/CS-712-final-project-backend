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


def rgb_to_hsi_global(rgb):
    with np.errstate(divide='ignore', invalid='ignore'):
        rgb = np.float32(rgb) / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        intensity = np.mean(rgb, axis=2)

        min_value = np.min(rgb, axis=2)
        saturation = 1 - 3 * min_value / (r + g + b + 1e-6)

        # Compute hue
        hue = np.arccos((0.5 * ((r - g) + (r - b)) / np.sqrt((r - g) ** 2 + (r - b) * (g - b) + 1e-6)))
        hue[b > g] = 2 * np.pi - hue[b > g]
        hue /= 2 * np.pi

        return np.stack([hue, saturation, intensity], axis=-1)


def hsi_to_rgb_global(hsi):
    h, s, i = hsi[..., 0], hsi[..., 1], hsi[..., 2]
    h *= 2 * np.pi
    x = i * (1 - s)
    m = i * (1 + s * np.cos(h) / np.cos(np.pi / 3 - h % (2 * np.pi / 3)))
    m[m > 1] = 1

    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
    idx = (h < 2 * np.pi / 3)
    b[idx], g[idx], r[idx] = x[idx], m[idx], 3 * i[idx] - (x[idx] + m[idx])
    idx = (h >= 2 * np.pi / 3) & (h < 4 * np.pi / 3)
    r[idx], b[idx], g[idx] = x[idx], m[idx], 3 * i[idx] - (x[idx] + m[idx])
    idx = (h >= 4 * np.pi / 3)
    g[idx], r[idx], b[idx] = x[idx], m[idx], 3 * i[idx] - (x[idx] + m[idx])

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 1) * 255


def equalize_intensity(hsi):
    i = hsi[..., 2]
    hist, bins = np.histogram(i.flatten(), bins=256, range=(0, 1))
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    equalized_intensity = np.interp(i.flatten(), bins[:-1], cdf_normalized / 255)
    hsi[..., 2] = equalized_intensity.reshape(i.shape)
    return hsi
