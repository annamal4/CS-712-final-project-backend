from PIL import Image

def convert_png_to_pgm(png_filename, pgm_filename):
    # Open the PNG image using Pillow
    png_image = Image.open(png_filename)

    # Convert the PNG image to grayscale (L mode)
    gray_image = png_image.convert('L')

    # Save the grayscale image as PGM
    gray_image.save(pgm_filename)

# Example usage
png_filename = 'test-image.png'
pgm_filename = 'output.pgm'
convert_png_to_pgm(png_filename, pgm_filename)