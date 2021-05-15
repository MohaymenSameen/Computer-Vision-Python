import glob
from os import path
import cv2 as cv
from PIL import ImageFilter
from PIL import Image
from matplotlib import pyplot as plt
from skimage.util import random_noise


def noisetype(noise_type, image):
    img = cv.imread(image)

    if noise_type == "gauss":
        gauss = random_noise(img, mode='gaussian', seed=None, clip=True)
        plt.imshow(gauss[..., ::-1]), plt.title('Gaussian')
        plt.savefig('(%s)_gaussian.jpg' % image, dpi=300)
        plt.show()
    elif noise_type == "s&p":
        sp = random_noise(img, mode='s&p', seed=None, clip=True)
        plt.imshow(sp[..., ::-1]), plt.title('Salt & Pepper')
        plt.savefig('(%s)_s&p.jpg' % image, dpi=300)
        plt.show()
    elif noise_type == "poisson":
        poiss = random_noise(img, mode='poisson', seed=None, clip=True)
        plt.imshow(poiss[..., ::-1]), plt.title('Poisson')
        plt.savefig('(%s)_poisson.jpg' % image, dpi=300)
        plt.show()
    elif noise_type == "spectacle":
        spec = random_noise(img, mode='speckle', seed=None, clip=True)
        plt.imshow(spec[..., ::-1]), plt.title('Spectacle')
        plt.savefig('(%s)_spectacle.jpg' % image, dpi=300)
        plt.show()
    else:
        print("invalid noise type")

    return plt


def filter(filter_type):
    images = []
    for imagepath in glob.glob("mohaymen/mountain/*.jpg"):
        images.append(imagepath)
    count = 0
    outpath = "mohaymen/mountain/%s" % filter_type
    for img in images:
        img2 = cv.imread(img)
        if filter_type == "mean":
            new_image = cv.blur(img2, (9, 9))
            plt.imshow(new_image[..., ::-1]), plt.title('Mean')
            plt.savefig(path.join(outpath, "%d_mean.jpg" % count))
            plt.show()
            count += 1
        elif filter_type == "gaussblur":
            new_image = cv.GaussianBlur(img2, (9, 9), 0)
            plt.imshow(new_image[..., ::-1]), plt.title('Gaussian Blur')
            plt.savefig(path.join(outpath, "%d_gaussian_blur.jpg" % count))
            plt.show()
            count += 1
        elif filter_type == "median":
            new_image = cv.medianBlur(img2, 9)
            plt.imshow(new_image[..., ::-1]), plt.title('Median Filter')
            plt.savefig(path.join(outpath, "%d_median_filter.jpg" % count))
            plt.show()
            count += 1
        elif filter_type == "laplacian":
            new_image = cv.Laplacian(img2, cv.CV_64F)
            plt.imshow(img2 + new_image[..., ::-1]), plt.title('Laplacian')
            plt.savefig(path.join(outpath, "%d_laplacian.jpg" % count))
            plt.show()
            count += 1
        elif filter_type == "unsharp":
            img2 = img2[..., ::-1]
            img2 = Image.fromarray(img2.astype('uint8'))
            new_image = img2.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
            plt.imshow(new_image), plt.title('unsharp')
            plt.savefig(path.join(outpath, "%d_unsharp.jpg" % count))
            plt.show()
            count += 1
        else:
            print("invalid filter type")
    return plt


# noisetype("spectacle", "mountain.jpg")
filter("unsharp")
