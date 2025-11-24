import cv2
import matplotlib.pyplot as plt
import numpy as np

def preprocess_image(input_filename='image.png', output_filename='image_processed.png'):
    # Read image
    img = cv2.imread(input_filename, cv2.IMREAD_GRAYSCALE)
    
    # Filter image -> binary image
    thresh, img_filtered = cv2.threshold(img, np.unique(img)[1], 255, cv2.THRESH_BINARY_INV)

    # Obtain y coordinates of the upmost and bottommost black pixels
    vertical_boundaries = np.where(img_filtered.argmax(axis=1) > 0)[0]
    upper_y = vertical_boundaries.min()
    lower_y =  vertical_boundaries.max()

    # Obtain x coordinates of the upmost and bottommost black pixels
    horizontal_boundaries = np.where(img_filtered.argmax(axis=0) > 0)[0]
    upper_x = horizontal_boundaries.min()
    lower_x =  horizontal_boundaries.max()

    # Calculate x and y dimensions of the 0-frame image (the edges of the drawing are snapped to the frame of the image)
    x_dim = lower_x - upper_x
    y_dim = lower_y - upper_y

    # Calculate how many pixels to add to the meager dimension so the resulting image is (almost) a square
    # Append the (difference // 2)-pixel wide strip to both sides of the meager dimension
    complement_pixels = (np.max([x_dim, y_dim]) - np.min([x_dim, y_dim])) // 2

    if np.argmin([x_dim, y_dim]) == 0:
        lower_x = lower_x + complement_pixels
        upper_x = upper_x - complement_pixels
    else:
        lower_y = lower_y + complement_pixels
        upper_y = upper_y - complement_pixels

    # Reverse the colors in the image so the results looks like the input
    img_square = cv2.bitwise_not(img_filtered[upper_y:lower_y, upper_x:lower_x])

    # Resize image to 224 x 244 (ResNet mandated input size)
    img_resized = cv2.resize(img_square, (224, 224))

    # Save image
    cv2.imwrite(output_filename, img_resized)

    return output_filename
