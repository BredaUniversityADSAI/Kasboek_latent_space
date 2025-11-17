import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_heatmap(scribble_filename="image.png", heatmap_filename="heatmap.png"):
    '''
    Create an additional visualization that illustrates the distribution of black pixels in regions of the scribble image

    Params:
        scribble_filename: the name of the scribble image with filetype extension
        heatmap_filename: the name of the heatmap image with filetype extension
    
    Returns:
        heatmap_filename: the same heatmap filename as the parameter
    '''

    # Load the image (black & white)
    img = cv2.imread(f"{scribble_filename}", cv2.IMREAD_GRAYSCALE)

    # Convert to binary: 1 = black, 0 = white
    # You can adjust threshold if needed
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)

    # Split the image into a 3x3 grid
    h, w = binary.shape
    h_step, w_step = h // 3, w // 3

    heatmap = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            region = binary[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            black_ratio = np.mean(region)  # average = fraction of black pixels
            heatmap[i, j] = black_ratio

    # Normalize (optional, for visualization)
    heatmap /= heatmap.max()
    plt.xticks(np.arange(0, 3, 1))
    plt.yticks(np.arange(0, 3, 1))

    # Plot heatmap
    plt.imshow(heatmap, cmap='grey', interpolation='nearest')
    plt.axis('off')               # Hide axes and ticks
    plt.tight_layout(pad=0)       # Remove internal padding
    plt.margins(0)
    plt.savefig(f'{heatmap_filename}')

    return heatmap_filename