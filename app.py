from flask import Flask, request, send_file
import logging
from io import BytesIO
import numpy as np
import cv2
from localHist import localHistEqual4e
from PIL import Image

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_pgm_file(file):
    # Read header
    line1 = file.readline().decode().strip()
    line2 = file.readline().decode().strip()
    dimensions = file.readline().decode().strip().split()
    width = int(dimensions[0])
    height = int(dimensions[1])
    max_val = int(file.readline().decode().strip())

    # Read image data
    img_data = file.read()

    return line1, line2, width, height, max_val, img_data


def invert_pixels(img_data):
    inverted_img_data = bytearray([255 - byte for byte in img_data])
    return inverted_img_data


def compile_pgm_file(magic_number, line2, width, height, max_val, img_data):
    # Create an in-memory file
    output_file = BytesIO()
    output_file.write(f"{magic_number}\n".encode())
    output_file.write(f"{line2}\n".encode())
    output_file.write(f"{width} {height}\n".encode())
    output_file.write(f"{max_val}\n".encode())
    output_file.write(img_data)
    output_file.seek(0)  # Move the cursor to the beginning of the file

    return output_file


# Write data to disk
# Use for testing purpose
def write_pgm_file(filename, compiled_data):
    with open(filename, 'wb') as f:
        f.write(compiled_data.getvalue())


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

def global_histogram_equalization(image_data, width, height):
    image = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            image[i][j] = image_data[i * width + j]

    histogram = calculate_histogram(image, width, height)
    cumulative_histogram = calculate_cumulative_histogram(histogram)

    total_pixels = width * height
    normalized_histogram = [int(round((cumulative_histogram[i] / total_pixels) * 255)) for i in range(256)]

    equalized_image = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            equalized_image[i][j] = normalized_histogram[image[i][j]]

    equalized_image_bytes = bytes(sum(equalized_image, []))

    return equalized_image_bytes


@app.route('/image-local', methods=['POST'])
def process_image():
    logger.info("Processing image with local histogram equalization")

    if 'image' not in request.files:
        return 'No image file in the request', 400

    image_file = request.files['image']

    input_image = np.array(Image.open(image_file))
    output = localHistEqual4e(input_image, 3, 3)

    output_image = Image.fromarray(output)
    output_data = BytesIO()
    output_image.save(output_data, format='PNG')
    output_data.seek(0)

    return send_file(output_data, mimetype='image/png', as_attachment=True,
                     download_name='local_histogram_equalized_image.png')


@app.route('/image-global', methods=['POST'])
def process_global_image():
    logger.info("Processing image with global histogram equalization")
    if 'image' not in request.files:
        return 'No image file in the request', 400
    image_file = request.files['image']
    if image_file.filename.endswith('.pgm'):
        line1, line2, width, height, max_val, img_data = read_pgm_file(image_file.stream)

        # apply global histogram equalization
        global_equalized_img_data = global_histogram_equalization(img_data, width, height)
        global_output_data = compile_pgm_file(line1, line2, width, height, max_val, global_equalized_img_data)
        global_output_filename = 'global_equalized_image.pgm'
        write_pgm_file(global_output_filename, global_output_data)
        return send_file(global_output_data, mimetype='image/pgm', as_attachment=True,
                         download_name=global_output_filename)
    else:
        return 'Invalid file format. Only PGM files are allowed.'


if __name__ == '__main__':
    app.run(debug=True)
