import io

from flask import Flask, request, send_file
import logging
from io import BytesIO
import numpy as np

from globalHist import globalHistEqual, rgb_to_hsi_global, equalize_intensity, hsi_to_rgb_global
from localHist import local_hist_equalization, local_hist_color
from PIL import Image


app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/image-local', methods=['POST'])
def process_local_image():
    logger.info("Processing image with local histogram equalization")

    if 'image' not in request.files:
        return 'No image file in the request', 400

    image_file = request.files['image']

    input_image = np.array(Image.open(image_file))
    output = local_hist_equalization(input_image, 3, 3)

    output_image = Image.fromarray(output)
    output_image.save("processed_image.png")
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

    input_image = np.array(Image.open(image_file))
    output = globalHistEqual(input_image)

    output_image = Image.fromarray(output)
    output_image.save("global_hist.png")

    output_data = BytesIO()
    output_image.save(output_data, format='PNG')
    output_data.seek(0)

    return send_file(output_data, mimetype='image/png', as_attachment=True,
                     download_name='global_histogram_equalized_image.png')


@app.route('/image-compare', methods=['POST'])
def process_image_comparison():
    logger.info("Comparing image with local and global histogram equalization")
    if 'image' not in request.files:
        return 'No image file in the request', 400
    image_file = request.files['image']

    input_image = np.array(Image.open(image_file))

    local_output = local_hist_equalization(input_image, 3, 3)
    global_output = globalHistEqual(input_image)

    diff_mask = np.abs(local_output - global_output)
    overlay_image = np.stack([input_image, input_image, input_image], axis=-1)
    overlay_image[diff_mask > 100] = [255, 0, 0]

    output_image = Image.fromarray(overlay_image)
    output_data = BytesIO()
    output_image.save(output_data, format='PNG')
    output_data.seek(0)

    return send_file(output_data, mimetype='image/png', as_attachment=True,
                     download_name='histogram-compared.png')


@app.route('/image-local-color', methods=['POST'])
def process_image_color():
    logger.info("Processing image with local histogram equalization")

    if 'image' not in request.files:
        return 'No image file in the request', 400

    image_file = request.files['image']

    input_image = np.array(Image.open(image_file))
    output = local_hist_color(input_image)

    output_image = Image.fromarray(output)
    output_image.save("processed_image.png")
    output_data = BytesIO()
    print("saving file")
    output_image.save(output_data, format='PNG')
    output_data.seek(0)

    return send_file(output_data, mimetype='image/png', as_attachment=True,
                     download_name='local_histogram_equalized_image.png')


@app.route('/image-global-color', methods=['POST'])
def image_global_color():
    # Check if the post request has the image part
    if 'image' not in request.files:
        return "No image part", 400
    image_file = request.files['image']
    if image_file.filename == '':
        return "No selected image", 400
    if image_file:
        # Read the image file
        img = Image.open(image_file.stream).convert('RGB')
        rgb = np.array(img)
        hsi = rgb_to_hsi_global(rgb)
        hsi = equalize_intensity(hsi)
        processed_rgb = hsi_to_rgb_global(hsi)
        processed_image = Image.fromarray(np.uint8(processed_rgb))

        # Save processed image to a bytes buffer
        img_byte_arr = io.BytesIO()
        processed_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return send_file(img_byte_arr, mimetype='image/png')

    return "Unsupported image type", 400


if __name__ == '__main__':
    app.run(debug=True)
