import matplotlib.pyplot as plt

def border_image(image, border_pixels, width):
    '''
    This Function takes a binary image and a list with the position of the border pixels of the image
    and plots an image with cell borders of custom width.
    To recieve the border pixels use 'find_border' in metrics

    :param image: Ground truth image
    :param border_pixels: List of border pixels of input image
    :param width: border width
    :return:Plots the imag with cell borders
    '''

    half_width = width//2

    border_image = image.copy()

    for pixel in border_pixels:
        for a in range(-half_width, half_width+1):
            for b in range(-half_width, half_width+1):
                if pixel[0] + a < border_image.shape[0] and pixel[1] + b < border_image.shape[1]:
                    border_image[pixel[0] + a][pixel[1] + b] = 120

    plt.imshow(border_image, 'PiYG')
    plt.show()

def overlay (test_image, gorund_thruth):
