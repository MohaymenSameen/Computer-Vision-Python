import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def image_channels(image):
    img = cv.imread(image)
    img = img[..., ::-1]
    blue_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]

    blue_img = np.zeros(img.shape)
    green_img = np.zeros(img.shape)
    red_img = np.zeros(img.shape)
    blue_img[:, :, 0] = blue_channel
    green_img[:, :, 1] = green_channel
    red_img[:, :, 2] = red_channel

    plt.imshow(blue_img)
    plt.savefig("mohaymen/mountain_red_img.jpg", dpi=300)
    plt.show()
    return plt


image_channels("mohaymen/mountain/mountain.jpg")
