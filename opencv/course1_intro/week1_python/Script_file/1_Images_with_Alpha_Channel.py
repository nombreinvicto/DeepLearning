# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Path of the PNG image to be loaded
imagePath = DATA_PATH + "/images/panther.png"

# Read the image
# Note that we are passing flag = -1 while reading the image ( it will read the image as is)
imgPNG = cv2.imread(imagePath,-1)
print("image Dimension ={}".format(imgPNG.shape))

#First 3 channels will be combined to form BGR image
#Mask is the alpha channel of the original image
imgBGR = imgPNG[:,:,0:3]
imgMask = imgPNG[:,:,3]

cv2.imwrite("results/colorChannels.png",imgBGR)
cv2.imwrite("results/alphaChannel.png",imgMask)


cv2.imshow("Color Channel",imgBGR)
cv2.imshow("Alpha Channel",imgMask)
cv2.imshow("Original Image",imgPNG)

while True:
    c = cv2.waitKey(20)
    if c == 27:
        break

cv2.destroyAllWindows()