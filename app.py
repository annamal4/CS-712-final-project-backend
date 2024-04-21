from flask import Flask, request, send_file
import logging
from io import BytesIO

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

@app.route('/image', methods=['POST'])
def process_image():
    logger.info("test")

    if 'image' not in request.files:
        return 'No image file in the request', 400

    image_file = request.files['image']

    if image_file.filename.endswith('.pgm'):
        
        line1, line2, width, height, max_val, img_data = read_pgm_file(image_file)

        # Do image processing here
        inverted_img_data = invert_pixels(img_data) #why this?

        output_data = compile_pgm_file(line1, line2, width, height, max_val, inverted_img_data)

        # Filename for the output file
        output_filename = 'inverted_image.pgm'

        # Write data to disk (use for testing purpose only)
        write_pgm_file(output_filename, output_data)

        return send_file(output_data, mimetype='image/pgm', as_attachment=True, download_name=output_filename)
    else:
        return 'Invalid file format. Only PGM files are allowed.'

if __name__ == '__main__':
    app.run(debug=True)