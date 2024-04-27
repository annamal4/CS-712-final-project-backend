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


# global histogram equalization

def global_histogram_equalization(image_data, width, height):
    # Convert bytes to NumPy array
    image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width))

    # Apply global histogram equalization
    equalized_image = cv2.equalizeHist(image)

    return equalized_image


# local histogram equalization
def local_histogram_equalization(image_data, width, height, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convert bytes to NumPy array
    image = np.frombuffer(image_data, dtype=np.uint8)
    image = image.reshape((height, width))  # Use the height and width from the parameters

    # Create a CLAHE object (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE to the grayscale image
    equalized_image = clahe.apply(image)

    return equalized_image


@app.route('/image-local', methods=['POST'])
def process_image():
    logger.info("test")

    if 'image' not in request.files:
        return 'No image file in the request', 400

    image_file = request.files['image']

    if image_file.filename.endswith('.pgm'):

        line1, line2, width, height, max_val, img_data = read_pgm_file(image_file.stream)

        # Apply local histogram equalization
        local_equalized_img_data = local_histogram_equalization(img_data, width, height, clip_limit=2.0,
                                                                tile_grid_size=(8, 8))

        local_output_data = compile_pgm_file(line1, line2, width, height, max_val, local_equalized_img_data.tobytes())

        local_output_filename = 'local_equalized_image.pgm'
        write_pgm_file(local_output_filename, local_output_data)

        return send_file(local_output_data, mimetype='image/pgm', as_attachment=True,
                         download_name=local_output_filename)
    else:
        return 'Invalid file format. Only PGM files are allowed.'


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
        global_output_data = compile_pgm_file(line1, line2, width, height, max_val, global_equalized_img_data.tobytes())
        global_output_filename = 'global_equalized_image.pgm'
        write_pgm_file(global_output_filename, global_output_data)
        return send_file(global_output_data, mimetype='image/pgm', as_attachment=True,
                         download_name=global_output_filename)
    else:
        return 'Invalid file format. Only PGM files are allowed.'


@app.route('/local-histogram', methods=['POST'])
def process_local_histogram():
    logger.info("Processing image with local histogram equalization")

    if 'image' not in request.files:
        return 'No image file in the request', 400

    image_file = request.files['image']

    if image_file.filename.endswith('.png'):
        input_image = np.array(Image.open(image_file))
        output = localHistEqual4e(input_image, 3, 3)

        output_image = Image.fromarray(output)
        output_data = BytesIO()
        output_image.save(output_data, format='PNG')
        output_data.seek(0)

        return send_file(output_data, mimetype='image/png', as_attachment=True,
                         download_name='local_histogram_equalized_image.png')
    else:
        return 'Invalid file format. Only PNG files are allowed.'


if __name__ == '__main__':
    app.run(debug=True)
