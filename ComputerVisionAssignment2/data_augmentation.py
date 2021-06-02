import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import os
import cv2
#%matplotlib inline

generalpath = "/Users/mohaymen/Documents/GitHub/Computer-Vision-Python/ComputerVisionAssignment2/all_test_tubes/"
inputpath = os.path.join(generalpath, 'Dset')
outputpath = os.path.join(generalpath, 'TT')

imageCountOfLetter = 0

for subdir, dirs, files in os.walk(inputpath):
    for filename in files:
        if filename.endswith('.jpg'):
            imgpath = os.path.join(subdir, filename)
            filename_stripped = filename.strip('.jpg')
            outputfilepath = os.path.join(subdir, filename_stripped).replace('v1', 'v2 (augmented)')

            print(imgpath)
            print(filename_stripped)
            print(outputfilepath)

            # 1 Default img
            defaultimg = imageio.imread(imgpath)
            # ia.imshow(img)
            img1 = cv2.resize(defaultimg, (690, 920))
            img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            cv2.imwrite(outputfilepath + ".jpg", img)

            # 2 Rotate
            rotate = iaa.Affine(rotate=(-30, 30))
            rotated_image = rotate.augment_image(img)
            # ia.imshow(rotated_image)
            cv2.imwrite(outputfilepath + "_rotated.jpg", rotated_image)

            # 3 Gauss noise
            gaussian_noise = iaa.AdditiveGaussianNoise(10, 20)
            noise_image = gaussian_noise.augment_image(img)
            # ia.imshow(noise_image)
            cv2.imwrite(outputfilepath + "_gauss_noise.jpg", noise_image)

            # 4 Cropping
            crop = iaa.Crop(percent=(0, 0.3)) # crop image
            crop_image = crop.augment_image(img)
            ia.imshow(crop_image)
            cv2.imwrite(outputfilepath + "_crop.jpg", crop_image)

            # 5 Shearing
            shear = iaa.Affine(shear=(5))
            shear_image = shear.augment_image(img)
            ia.imshow(shear_image)
            cv2.imwrite(outputfilepath + "_shear.jpg", shear_image)

            # 6 Flipping horizontally
            flip_hr = iaa.Fliplr(p=1.0)
            flip_hr_image = flip_hr.augment_image(img)
            # ia.imshow(flip_hr_image)
            cv2.imwrite(outputfilepath + "_flip_hr.jpg", flip_hr_image)

            # 7 Flip vertically
            flip_vr = iaa.Flipud(p=1.0)
            flip_vr_image= flip_vr.augment_image(img)
            ia.imshow(flip_vr_image)
            cv2.imwrite(outputfilepath + "_flip_vr.jpg", flip_vr_image)

            # 8 Change brightness
            contrast = iaa.GammaContrast(gamma=2.0)
            contrast_image = contrast.augment_image(img)
            # ia.imshow(contrast_image)
            cv2.imwrite(outputfilepath + "_constrast.jpg", contrast_image)

            # 9 Scaling image
            scale_im = iaa.Affine(scale={"x": (1.5, 1.0), "y": (1.5, 1.0)})
            scale_image = scale_im.augment_image(img)
            ia.imshow(scale_image)
            cv2.imwrite(outputfilepath + "_scaled.jpg", scale_image)

            # 10 Blur image
            blurred_img = cv2.blur(img, (5, 5))
            # ia.imshow(blurred_img)
            cv2.imwrite(outputfilepath + "_blurred.jpg", blurred_img)


