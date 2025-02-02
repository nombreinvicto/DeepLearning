# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Read image
image = cv2.imread(DATA_PATH+"images/boy.jpg")

brightnessOffset = 50

# Add the offset for increasing brightness
brightHigh = image + brightnessOffset

cv2.imwrite("../results/highBrightness.png",brightHigh)

print("Original Image Datatype : {}".format(image.dtype))
print("Brightness Image Datatype : {}\n".format(brightHigh.dtype))

print("Original Image Highest Pixel Intensity : {}".format(image.max()))
print("Brightness Image Highest Pixel Intensity : {}".format(brightHigh.max()))

cv2.imshow("Original Image",image)
cv2.imshow("Brightness Image",brightHigh)

while True:
    c = cv2.waitKey(20)
    if c == 27:
        break

cv2.destroyAllWindows()
