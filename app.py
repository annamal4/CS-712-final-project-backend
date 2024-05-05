from flask import Flask, request, send_file
import logging
from io import BytesIO
import numpy as np
from globalHist import globalHistEqual
from localHist import localHistEqual4e
from PIL import Image

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def invert_pixels(img_data):
    inverted_img_data = bytearray([255 - byte for byte in img_data])
    return inverted_img_data

@app.route('/image-local', methods=['POST'])
def process_local_image():
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

    input_image = np.array(Image.open(image_file))
    output = globalHistEqual(input_image)

    output_image = Image.fromarray(output)
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

    local_output = localHistEqual4e(input_image, 3, 3)
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


if __name__ == '__main__':
    app.run(debug=True)
