import matplotlib.pyplot as plt

def border_image(image, border_pixels, width):

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
