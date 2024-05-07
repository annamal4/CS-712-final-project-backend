import cv2
import numpy as np


def histogram_equalization(img):
    # Calculate the histogram
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    # Calculate the cumulative distribution function
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()  # Normalization

    # Normalize the CDF
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Use the CDF to remap the original gray levels in the image to the equalized levels
    img_equalized = cdf[img]

    return img_equalized

def local_histogram_equalization(img, block_size=256):
    if block_size <= 0:
        raise ValueError("Block size must be a positive integer.")

    # Create an empty image to store the local equalized result
    img_local_equalized = np.zeros_like(img)

    # Iterate over blocks of the image and apply histogram equalization
    for i in range(0, img.shape[0], block_size):
        for j in range(0, img.shape[1], block_size):
            block = img[i:i+block_size, j:j+block_size]
            equalized_block = histogram_equalization(block)
            img_local_equalized[i:i+block_size, j:j+block_size] = equalized_block

    return img_local_equalized


def process_video(input_path, output_path, block_size=256):
    # Open the video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4
    out = cv2.VideoWriter(output_path + r'\equalized_video.mp4', fourcc, fps, (frame_width, frame_height), isColor=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization
        equalized = histogram_equalization(gray)

        # Convert back to BGR
        equalized_color = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

        # Ensure the frame size matches the output settings
        if (equalized_color.shape[1] != frame_width) or (equalized_color.shape[0] != frame_height):
            equalized_color = cv2.resize(equalized_color, (frame_width, frame_height))

        # Write the frame
        out.write(equalized_color)

        # Display frame; remove if unwanted
        # cv2.imshow('Frame', equalized_color)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release everything when job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete. Output saved to:", output_path + r'\equalized_video.mp4')


# Usage
# input_video_path = r"C:\Users\nagal\Downloads\4999917-uhd_3840_2160_30fps.mp4"
# output_video_path = r"C:\Users\nagal\Downloads"
input_video_path = r"./ultrasound-test-video.mp4"
output_video_path = r"./"
process_video(input_video_path, output_video_path)

