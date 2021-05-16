import cv2
import numpy as np
from cv2 import COLOR_BGR2LAB, COLOR_RGB2Lab
from matplotlib import pyplot as plt

""""
img = cv2.imread('wiki.jpg',0)

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]
"""

image = cv2.imread("mohaymen/Low contrast Image.jpg")
lab = cv2.cvtColor(image, COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
plt.hist(l.flat, bins=20000, range=(0, 255))
newL = cv2.equalizeHist(l)
merge = cv2.merge([newL, a, b])
convert = cv2.cvtColor(merge, COLOR_RGB2Lab)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(l)
newimg = cv2.merge([cl1, a, b])
convert2 = cv2.cvtColor(newimg, COLOR_RGB2Lab)
plt.imshow(convert2)
plt.show()
