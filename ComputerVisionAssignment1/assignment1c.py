import cv2
from cv2 import COLOR_BGR2LAB, COLOR_RGB2Lab
from matplotlib import pyplot as plt


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
plt.savefig("mohaymen/high_contrast.jpg")
plt.show()
